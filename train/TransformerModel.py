import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _enc_block(in_ch, out_ch, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(out_ch),
    )


def _dec_block(in_ch, out_ch, last=False):
    if last:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_ch),
    )


class TransformerModel(nn.Module):
    """
    U-Net style CNN + Transformer encoder-decoder for occupancy grid prediction.

    CNN encoder downsamples each frame with skip connections saved at each level.
    A transformer encoder-decoder processes the bottleneck tokens.  The CNN
    decoder upsamples back to full resolution, concatenating skip features
    (averaged across input timesteps) at each level for sharp reconstruction.

    Input:  (B, T, 2, H, W)   -- T occupancy+delta grids
            (B, T, 2)         -- per-frame motion metadata
    Output: (B, F, 1, H, W)   -- F predicted occupancy grids
    """

    ENC_CHANNELS = [2, 32, 64, 128, 256]

    def __init__(self, grid_h: int = 201, grid_w: int = 201,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 4,
                 n_input: int = 5, n_target: int = 5,
                 num_decoder_layers: int = 2,
                 motion_dim: int = 2):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.d_model = d_model
        self.n_input = n_input
        self.n_target = n_target
        self.motion_dim = motion_dim

        ch = self.ENC_CHANNELS

        # --- CNN encoder layers (separate so we can grab skip features) ---
        self.enc1 = _enc_block(ch[0], ch[1], kernel_size=5, stride=2, padding=2)  # 201→101
        self.enc2 = _enc_block(ch[1], ch[2])                                       # 101→51
        self.enc3 = _enc_block(ch[2], ch[3])                                       # 51→26
        self.enc4 = _enc_block(ch[3], ch[4])                                       # 26→13
        self.enc5 = _enc_block(ch[4], d_model)                                     # 13→7

        # determine spatial size after downsampling
        with torch.no_grad():
            dummy = torch.zeros(1, self.ENC_CHANNELS[0], grid_h, grid_w)
            d = self.enc1(dummy)
            d = self.enc2(d)
            d = self.enc3(d)
            d = self.enc4(d)
            d = self.enc5(d)
            self.feat_h = d.shape[2]
            self.feat_w = d.shape[3]
        self.n_spatial = self.feat_h * self.feat_w

        # --- positional embeddings ---
        self.spatial_pe = nn.Parameter(torch.zeros(1, self.n_spatial, d_model))
        self.temporal_pe_enc = nn.Parameter(torch.zeros(1, n_input, 1, d_model))
        self.temporal_pe_dec = nn.Parameter(torch.zeros(1, n_target, 1, d_model))
        nn.init.normal_(self.spatial_pe, std=0.02)
        nn.init.normal_(self.temporal_pe_enc, std=0.02)
        nn.init.normal_(self.temporal_pe_dec, std=0.02)

        self.motion_mlp = nn.Sequential(
            nn.Linear(motion_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # --- transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- transformer decoder ---
        self.query_tokens = nn.Parameter(torch.zeros(1, self.n_spatial, d_model))
        nn.init.normal_(self.query_tokens, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # --- CNN decoder with skip connections ---
        # Each stage receives upsampled features + skip features (concatenated along channels)
        # 7→14: d_model + 256(skip) → 256
        self.dec5 = _dec_block(d_model, ch[4])
        self.skip_conv4 = _enc_block(ch[4] + ch[4], ch[4], kernel_size=3, stride=1, padding=1)

        # 14→28: 256 + 128(skip) → 128
        self.dec4 = _dec_block(ch[4], ch[3])
        self.skip_conv3 = _enc_block(ch[3] + ch[3], ch[3], kernel_size=3, stride=1, padding=1)

        # 28→56: 128 + 64(skip) → 64
        self.dec3 = _dec_block(ch[3], ch[2])
        self.skip_conv2 = _enc_block(ch[2] + ch[2], ch[2], kernel_size=3, stride=1, padding=1)

        # 56→112: 64 + 32(skip) → 32
        self.dec2 = _dec_block(ch[2], ch[1])
        self.skip_conv1 = _enc_block(ch[1] + ch[1], ch[1], kernel_size=3, stride=1, padding=1)

        # 112→224: 32 → 1 (no skip at full res)
        self.dec1 = _dec_block(ch[1], 1, last=True)

    def _encode_frames(self, x):
        """Run CNN encoder on all frames, returning bottleneck + skip features.

        x: (B*T, C, H, W)
        Returns: bottleneck (B*T, d_model, fh, fw), skips list [(B*T, ch, h, w), ...]
        """
        s1 = self.enc1(x)    # (B*T, 32, 101, 101)
        s2 = self.enc2(s1)   # (B*T, 64, 51, 51)
        s3 = self.enc3(s2)   # (B*T, 128, 26, 26)
        s4 = self.enc4(s3)   # (B*T, 256, 13, 13)
        s5 = self.enc5(s4)   # (B*T, d_model, 7, 7)
        return s5, [s1, s2, s3, s4]

    def _decode_with_skips(self, feat, skips):
        """Run CNN decoder with skip connections.

        feat: (B*F, d_model, fh, fw)
        skips: [s1, s2, s3, s4] each (B*F, ch, h, w)
        Returns: (B*F, 1, H', W')
        """
        s1, s2, s3, s4 = skips

        x = self.dec5(feat)                                           # → (B*F, 256, 14, 14)
        x = self._cat_and_conv(x, s4, self.skip_conv4)               # concat + conv

        x = self.dec4(x)                                              # → (B*F, 128, 28, 28)
        x = self._cat_and_conv(x, s3, self.skip_conv3)

        x = self.dec3(x)                                              # → (B*F, 64, 56, 56)
        x = self._cat_and_conv(x, s2, self.skip_conv2)

        x = self.dec2(x)                                              # → (B*F, 32, 112, 112)
        x = self._cat_and_conv(x, s1, self.skip_conv1)

        x = self.dec1(x)                                              # → (B*F, 1, 224, 224)
        return x

    @staticmethod
    def _cat_and_conv(upsampled, skip, conv):
        """Crop/pad skip to match upsampled spatial dims, concat, then conv."""
        dh = upsampled.shape[2] - skip.shape[2]
        dw = upsampled.shape[3] - skip.shape[3]
        if dh != 0 or dw != 0:
            skip = F.pad(skip, (0, dw, 0, dh))
        return conv(torch.cat([upsampled, skip], dim=1))

    def _encode_bottleneck_tokens(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Encode feature frames into bottleneck tokens.

        frame_features: (B, T, C, H, W)
        Returns: (B, T, n_spatial, d_model)
        """
        bsz, n_frames, channels, height, width = frame_features.shape
        flat_frames = frame_features.reshape(bsz * n_frames, channels, height, width)
        bottleneck, _ = self._encode_frames(flat_frames)
        bottleneck = bottleneck.reshape(bsz, n_frames, self.d_model, self.n_spatial)
        return bottleneck.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _build_occ_delta_features(frames: torch.Tensor, first_prev: torch.Tensor) -> torch.Tensor:
        """Build two-channel occupancy features from a 1-channel sequence."""
        prev_frames = torch.cat([first_prev, frames[:, :-1]], dim=1)
        delta = frames - prev_frames
        return torch.cat([frames, delta], dim=2)

    def forward(
        self,
        x_grids: torch.Tensor,
        x_motion: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B, T, C, H, W = x_grids.shape
        Ft = self.n_target

        if x_motion is None:
            x_motion = torch.zeros(B, T, self.motion_dim, device=x_grids.device, dtype=x_grids.dtype)

        # --- CNN encode each input frame, collecting skips ---
        x_flat = x_grids.reshape(B * T, C, H, W)
        bottleneck, skips = self._encode_frames(x_flat)  # bottleneck: (B*T, d_model, fh, fw)

        # --- use last input frame's skip features (same coord frame as targets) ---
        last_skips = []
        for s in skips:
            s_5d = s.reshape(B, T, *s.shape[1:])       # (B, T, ch, h, w)
            s_last = s_5d[:, -1]                        # (B, ch, h, w)
            last_skips.append(s_last.unsqueeze(1).expand(B, Ft, -1, -1, -1)
                              .reshape(B * Ft, *s_last.shape[1:]))  # (B*F, ch, h, w)

        # --- flatten bottleneck to tokens ---
        feat = bottleneck.reshape(B, T, self.d_model, self.n_spatial)
        tokens = feat.permute(0, 1, 3, 2).contiguous()   # (B, T, n_spatial, d_model)
        motion_embed = self.motion_mlp(x_motion.reshape(B * T, self.motion_dim)).reshape(B, T, 1, self.d_model)

        # --- add factored positional embeddings ---
        tokens = tokens + self.spatial_pe
        tokens = tokens + self.temporal_pe_enc[:, :T]
        tokens = tokens + motion_embed
        tokens = tokens.reshape(B, T * self.n_spatial, self.d_model)

        # --- transformer encode ---
        memory = self.encoder(tokens)                     # (B, T*S, d_model)

        # --- optionally encode GT future frames for teacher forcing ---
        target_tokens = None
        if targets is not None and teacher_forcing_ratio > 0.0:
            target_features = self._build_occ_delta_features(targets, x_grids[:, -1:, :1])
            target_tokens = self._encode_bottleneck_tokens(target_features)

        # --- autoregressive transformer decode ---
        decoded_steps = []
        prev_tokens = []
        for step_idx in range(Ft):
            step_query = self.query_tokens + self.spatial_pe
            step_query = step_query + self.temporal_pe_dec[:, step_idx]
            step_query = step_query.expand(B, -1, -1)

            if prev_tokens:
                tgt = torch.cat(prev_tokens + [step_query], dim=1)
            else:
                tgt = step_query

            step_out = self.decoder(tgt, memory)[:, -self.n_spatial:, :]
            decoded_steps.append(step_out)

            use_teacher = (
                target_tokens is not None
                and step_idx < Ft - 1
                and torch.rand((), device=x_grids.device) < teacher_forcing_ratio
            )
            prev_tokens.append(target_tokens[:, step_idx:step_idx + 1].reshape(B, self.n_spatial, self.d_model) if use_teacher else step_out)

        out = torch.stack(decoded_steps, dim=1)           # (B, F, S, d_model)
        out = out.reshape(B * Ft, self.n_spatial, self.d_model)
        out = out.permute(0, 2, 1).contiguous()
        out = out.reshape(B * Ft, self.d_model, self.feat_h, self.feat_w)

        # --- CNN decode with skip connections ---
        out = self._decode_with_skips(out, last_skips)     # (B*F, 1, H', W')
        out = out[:, :, :H, :W]
        out = out.reshape(B, Ft, 1, H, W)
        return out
