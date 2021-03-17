import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch_scatter as ts

class AgentsMapFusion(nn.Module):

    def __init__(self, in_channels=32, out_channels=32):
        super(AgentsMapFusion, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=4, stride=1, padding=1, dilation=1)

        self.deconv2 = deconv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                              k_size=4, stride=2, padding=1)

    def forward(self, input_tensor):
        conv1 = self.conv1.forward(input_tensor)
        conv2 = self.conv2.forward(self.pool1.forward(conv1))
        conv3 = self.conv3.forward(self.pool2.forward(conv2))

        up2 = self.deconv2.forward(conv2)
        up3 = F.interpolate(conv3, scale_factor=5)

        features = conv1 + up2 + up3
        return features

class SpatialEncodeAgent(nn.Module):

    def __init__(self, device, pooling_size):
        super(SpatialEncodeAgent, self).__init__()
        self.device = device
        self.pooling_size = pooling_size

    def forward(self, batch_size, encode_coordinates, agent_encodings):
        channel = agent_encodings.shape[-1]
        pool_vector = agent_encodings.transpose(1, 0) # [C X D]

        init_map_ts = torch.zeros((channel, batch_size*self.pooling_size*self.pooling_size), device=self.device) # [C X B*H*W]
        out, _ = ts.scatter_min(src=pool_vector, index=encode_coordinates, out=init_map_ts) # [C X B*H*W]
        out, _ = ts.scatter_max(src=pool_vector, index=encode_coordinates, out=out) # [C X B*H*W]

        out = out.reshape((channel, batch_size, self.pooling_size, self.pooling_size)) # [C X B X H X W]
        out = out.permute((1, 0, 2, 3)) # [B X C X H X W]

        return out

class SpatialFetchAgent(nn.Module):

    def __init__(self, device):
        super(SpatialFetchAgent, self).__init__()
        self.device = device

    def forward(self, fused_grid, agent_encodings, fetch_coordinates):
        # Rearange the fused grid so that linearized index may be used.
        batch, channel, map_h, map_w = fused_grid.shape
        fused_grid = fused_grid.permute((0, 2, 3, 1)) # B x H x W x C
        fused_grid = fused_grid.reshape((batch*map_h*map_w, channel))

        fused_encodings = fused_grid[fetch_coordinates]
        final_encoding = fused_encodings + agent_encodings

        return final_encoding


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class deconv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs

class ResnetShallow(nn.Module):

    def __init__(self, dropout=0.5):  # Output Size: 30 * 30
        super(ResnetShallow, self).__init__()

        self.trunk = models.resnet18(pretrained=True)

        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.upscale4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 7, stride=4, padding=3, output_padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.shrink = conv2DBatchNormRelu(in_channels=384, n_filters=32,
                                          k_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image):
        x = self.trunk.conv1(image)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)  # /8 the size
        x3 = self.trunk.layer3(x2)  # 16
        x4 = self.trunk.layer4(x3)  # 32

        x3u = self.upscale3(x3)
        x4u = self.upscale4(x4)

        xall = torch.cat((x2, x3u, x4u), dim=1)
        xall = F.interpolate(xall, size=(30, 30))
        final = self.shrink(xall)

        output = self.dropout(final)

        return output

class ShallowCNN(nn.Module):

    def __init__(self, in_channels, dropout=0.5):  # Output Size: 30 * 30
        super(ShallowCNN, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, n_filters=16,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=16, n_filters=16,
                                          k_size=4, stride=1, padding=2, dilation=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = conv2DBatchNormRelu(in_channels=16, n_filters=32,
                                          k_size=5, stride=1, padding=2, dilation=1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image):

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)

        output = self.dropout(x)

        return output

