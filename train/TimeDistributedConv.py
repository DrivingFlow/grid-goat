import torch

class TimeDistributedConv(torch.nn.Module):
    def __init__(self, conv_block):
        super().__init__()
        self.conv = conv_block

    def forward(self, x):      # x: (B, T, C, H, W)

        device = next(self.conv.parameters()).device
        x = x.to(device)
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.conv(x)
        _, C2, H2, W2 = x.shape
        return x.reshape(B, T, C2, H2, W2)