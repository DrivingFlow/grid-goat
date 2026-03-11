import torch
import torch.nn.functional as F
import numpy as np
from TimeDistributedConv import TimeDistributedConv

class EncoderMLPModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.max_seq_len = 10000
    self.d_model = 512
    self.pos_embed = torch.nn.Parameter(torch.zeros(1,self.max_seq_len,self.d_model))
    torch.nn.init.normal_(self.pos_embed)
    self.downsampling_layer = TimeDistributedConv(self.downsampling_conv())
    self.upsampling_layer = self.upsampling_conv()
    self.encoder_layers = self.encoder()
    self.out_images = 20
    self.mlp = self.MLP(out_timesteps=self.out_images)
    

  def downsampling_conv(self):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32),

        torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),

        torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128),

        torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(256),

        torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(512),
    )
    
  def upsampling_conv(self):
      return torch.nn.Sequential(
          torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
          torch.nn.ReLU(),
          torch.nn.BatchNorm2d(256),

          torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
          torch.nn.ReLU(),
          torch.nn.BatchNorm2d(128),

          torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
          torch.nn.ReLU(),
          torch.nn.BatchNorm2d(64),

          torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
          torch.nn.ReLU(),
          torch.nn.BatchNorm2d(32),

          torch.nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
          torch.nn.Sigmoid(),
      )

  # def downsampling_conv(self):
  #   return torch.nn.Sequential(
  #       # torch.nn.Conv2d(in_channels = 4, out_channels=32, kernel_size= 5, padding=0, stride=2),
  #       torch.nn.Conv2d(in_channels = 1, out_channels=32, kernel_size= 5, padding=0, stride=2),
  #       torch.nn.ReLU(),
  #       torch.nn.BatchNorm2d(num_features=32),
  #       torch.nn.Conv2d(in_channels = 32, out_channels=64, kernel_size= 3, padding=0, stride=2),
  #       torch.nn.ReLU(),
  #       torch.nn.BatchNorm2d(num_features=64),
  #       torch.nn.Conv2d(in_channels = 64, out_channels=128, kernel_size= 3, padding=0, stride=2),
  #       torch.nn.ReLU(),
  #       torch.nn.BatchNorm2d(num_features=128),
  #       torch.nn.Conv2d(in_channels = 128, out_channels=256, kernel_size= 3, padding=0, stride=2),
  #       torch.nn.ReLU(),
  #       torch.nn.BatchNorm2d(num_features=256),
  #       # torch.nn.Conv2d(in_channels = 256, out_channels=512, kernel_size= 3, padding=0, stride=2),
  #       # torch.nn.ReLU(),
  #       # torch.nn.BatchNorm2d(num_features=512),
  #       # torch.nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size= 3, padding=0, stride=2),
  #       # torch.nn.ReLU(),
  #       # torch.nn.BatchNorm2d(num_features=1024),
  #       # torch.nn.Conv2d(in_channels = 1024, out_channels=2048, kernel_size= 3, padding=0, stride=2),
  #       # torch.nn.ReLU(),
  #       # torch.nn.BatchNorm2d(num_features=2048)
  #   )

  # def upsampling_conv(self):
  #     return torch.nn.Sequential(
  #         # torch.nn.Conv2d(in_channels = 2048, out_channels=1024, kernel_size= 3, padding=0, stride=2),
  #         # torch.nn.ReLU(),
  #         # torch.nn.BatchNorm2d(num_features=1024),
  #         # torch.nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size= 3, padding=0, stride=2),
  #         # torch.nn.ReLU(),
  #         # torch.nn.BatchNorm2d(num_features=512),
  #         torch.nn.Conv2d(in_channels = 512, out_channels=256, kernel_size= 3, padding=0, stride=2),
  #         torch.nn.ReLU(),
  #         torch.nn.BatchNorm2d(num_features=256),
  #         torch.nn.Conv2d(in_channels = 256, out_channels=128, kernel_size= 3, padding=0, stride=2),
  #         torch.nn.ReLU(),
  #         torch.nn.BatchNorm2d(num_features=128),
  #         torch.nn.ConvTranspose2d(in_channels = 128, out_channels=64, kernel_size= 3, padding= 0, stride=2),
  #         torch.nn.ReLU(),
  #         torch.nn.BatchNorm2d(num_features=64),
  #         torch.nn.ConvTranspose2d(in_channels = 64, out_channels=32, kernel_size= 3, padding= 0, stride=2),
  #         torch.nn.ReLU(),
  #         torch.nn.BatchNorm2d(num_features=32),
  #         torch.nn.ConvTranspose2d(in_channels = 32, out_channels=1, kernel_size= 7, padding= 0, stride=2, output_padding=1),
  #         torch.nn.Sigmoid()
  #     )

  # def upsampling_conv(self):
  #   return torch.nn.Sequential(
  #       # torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
  #       # torch.nn.ReLU(),
  #       # torch.nn.BatchNorm2d(256),

  #       torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
  #       torch.nn.ReLU(),
  #       torch.nn.BatchNorm2d(128),

  #       torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
  #       torch.nn.ReLU(),
  #       torch.nn.BatchNorm2d(64),

  #       torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
  #       torch.nn.ReLU(),
  #       torch.nn.BatchNorm2d(32),

  #       torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2, output_padding=1),
  #       torch.nn.Sigmoid(),
  #   )

  def preprocess_for_transformer(self, x):
      B, T, C, H, W = x.shape
      x = x.reshape(B, T, C, H*W)
      x = x.permute(0, 1, 3, 2)
      x = x.reshape(B, T*H*W, C)
      return x, B, T, C, H, W

  def post_transformer(self, x, B, T, C, H, W):
      x = x.reshape(B, T, H * W, C)
      x = x.permute(0, 1, 3, 2)
      x = x.reshape(B, T, C, H, W)
      x = x.reshape(B*T, C, H, W)
      return x

  def encoder(self):
    return torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=4*512, activation='relu', batch_first=True, norm_first=True), num_layers=2)
  
  def MLP(self, out_timesteps):
     return torch.nn.Linear(5, out_timesteps)

  def forward(self, x):

    B0, T0, C0, H0, W0 = x.shape

    # print(x.shape)

    x_down = self.downsampling_layer(x)
    x_transformer, B, T, C, H, W = self.preprocess_for_transformer(x_down)
    seq_len = x_transformer.size(1)
    x_transformer += self.pos_embed[:, :seq_len, :]

    x_out = self.encoder_layers(x_transformer)
    x_out = self.post_transformer(x_out, B, T, C, H, W)
    out = self.upsampling_layer(x_out)
    # print(out.shape)
    out = out[..., :H0, :W0]
    out = out.view(B0, T0, 1, H0, W0)

    out = out.permute(0, 3, 4, 2, 1)           # (B,H,W,1,5)
    out = self.mlp(out)                           # (B,H,W,1,n)
    out = out.permute(0, 4, 3, 1, 2)          
    # print(out.shape)

    return out[:, -self.out_images:]
