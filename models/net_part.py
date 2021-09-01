from torch import nn

class conv3d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv3d, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x
