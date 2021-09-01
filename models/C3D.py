import torch
from torch import nn
from models.net_part import *


class C3D(nn.Module):

    def __init__(self, num_classes=101, train_type='SSL'):
        super(C3D, self).__init__()
        self.num_classes = num_classes
        self.train_type = train_type

        self.conv1 = conv3d(3, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = conv3d(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3_1 = conv3d(128, 256)
        self.conv3_2 = conv3d(256, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4_1 = conv3d(256, 512)
        self.conv4_2 = conv3d(512, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5_1 = conv3d(512, 512)
        self.conv5_2 = conv3d(512, 512)
        self.pool5 = nn.AdaptiveAvgPool3d(1)
        self.linear_ssl_1 = nn.Linear(512, self.num_classes)
        self.linear_ssl_2 = nn.Linear(512, self.num_classes)
        self.linear_finetune = nn.Linear(512, self.num_classes)

    def subforward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.pool5(x)

        return x

    def forward(self, x):

        if self.train_type == 'Finetune':
            x = self.subforward(x)
            x = x.view(-1, x.size(1))
            x = self.linear_finetune(x)
            return x

        if self.train_type == 'SSL':
            # shape of concat tensor: batch, channel, frames, height, width
            # concat two tensor(rgb tensor and diff tensor) through frames-dimension
            # so chunk input tensor through frames-dimension
            t_list = x.chunk(2, dim=2)
            x_rgb = self.subforward(t_list[0])
            x_diff = self.subforward(t_list[1])
            x_rgb = x_rgb.view(-1, x_rgb.size(1))
            x_diff = x_diff.view(-1, x_diff.size(1))
            x_rgb = self.linear_ssl_1(x_rgb)
            x_diff = self.linear_ssl_2(x_diff)

            return x_rgb, x_diff


if __name__ == '__main__':
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 32, 112, 112))
    # c3d = C3D(num_classes=8, train_type='Finetune')
    c3d = C3D(num_classes=8, train_type='SSL')
    output_rgb, output_diff = c3d(input_tensor)

    print(output_rgb.shape)
    print(output_diff.shape)