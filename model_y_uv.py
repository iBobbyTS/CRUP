import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
            self,
            inc, outc,
            kernel_size=3, padding=1, stride=1,
            use_bias=True, activation=nn.ReLU, batch_norm=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super().__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        self.activation = activation() if activation else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, bilateral_grid, guidemap):
        # Nx12x8x16x16
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        hg = hg.to(self.device)
        wg = wg.to(self.device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()
        self.n_channel = n_channel

    def forward(self, coeff, full_res_input):
        """
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        """
        print(coeff[:, 4 * 0:4 * 0 + 3, :, :].shape, full_res_input.shape)
        print(coeff[:, 4 * 1:4 * 1 + 3, :, :].shape, full_res_input.shape)
        [full_res_input * coeff[:, 4 * i:4 * i + 3, :, :] for i in range(self.n_channel)]
        return torch.cat([torch.sum(
            full_res_input * coeff[:, 4*i:4*i+3, :, :], dim=1, keepdim=True
        ) + coeff[:, [4*i+3], :, :] for i in range(self.n_channel)], dim=1)


class GuideNN(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Sigmoid)  # nn.Tanh

    def forward(self, x):
        return self.conv2(self.conv1(x))  # .squeeze(1)


class Coeffs(nn.Module):
    def __init__(
            self,
            channels=2,
            nin=4, nout=3,
            luma_bins=8, channel_multiplier=1, spatial_bin=16, batch_norm=False, net_input_size=256
    ):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.luma_bins = luma_bins
        self.channel_multiplier = channel_multiplier
        self.spatial_bin = spatial_bin
        self.batch_norm = batch_norm
        self.net_input_size = net_input_size

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(self.net_input_size / self.spatial_bin))  # 4
        splat_layer_list = list(np.geomspace(
            channels,
            self.channel_multiplier * (2 ** (n_layers_splat - 1)) * self.luma_bins,
            n_layers_splat + 1
        ).round().astype(np.uint16))
        self.splat_features = nn.Sequential(*[ConvBlock(
            splat_layer_list[i], splat_layer_list[i + 1],
            stride=2, batch_norm=self.batch_norm if i > 0 else False
        ) for i in range(n_layers_splat)])

        # global_features_conv
        n_layers_global = int(np.log2(self.spatial_bin / 4))
        global_layer_list = list(np.geomspace(
            splat_layer_list[-1],
            self.channel_multiplier * 8 * self.luma_bins / 3 * channels,
            n_layers_global + 1
        ).round().astype(np.uint16))
        self.global_features_conv = nn.Sequential(*[
            ConvBlock(
                global_layer_list[i], global_layer_list[i + 1],
                3, stride=2, batch_norm=self.batch_norm
            ) for i in range(n_layers_global)
        ])

        # global_features_fc
        global_features_fc_list = list(np.geomspace(
            global_layer_list[-1] * 4 * 4,
            8 * self.channel_multiplier * self.luma_bins / 3 * channels,
            4
        ).round().astype(np.uint16))
        self.global_features_fc = nn.Sequential(*[
            FC(
                global_features_fc_list[i], global_features_fc_list[i + 1],
                batch_norm=self.batch_norm
            ) for i in range(3)
        ])

        # local features
        local_features_layer_list = list(np.geomspace(
            splat_layer_list[-1],
            8 * self.channel_multiplier * self.luma_bins / 3 * channels,
            3
        ).round().astype(np.uint16))
        self.local_features = nn.Sequential(
            ConvBlock(
                local_features_layer_list[0],
                local_features_layer_list[1],
                3, batch_norm=self.batch_norm
            ),
            ConvBlock(
                local_features_layer_list[1],
                local_features_layer_list[2],
                3, activation=None, use_bias=False
            )
        )

        # conv_out
        self.conv_out = ConvBlock(
            round(8 * self.channel_multiplier * self.luma_bins / 3 * channels),
            round(self.luma_bins * nout * nin / 3 * channels),
            1, padding=0, activation=None
        )

        # Final
        self.split_and_stack = lambda x: torch.stack(
            torch.split(x, int(self.luma_bins * nout * nin / 3 * channels / 8), 1), 2)

    def forward(self, lowres_input):
        bs = lowres_input.shape[0]
        # splat_features
        x = self.splat_features(lowres_input)
        splat_features = self.local_features(x)
        # global_features
        x = self.global_features_conv(x)
        x = x.view(bs, -1)
        x = self.global_features_fc(x)
        # ReLU
        fusion = self.relu(
            splat_features + x.view(bs, x.shape[1], 1, 1)
        )
        # Final
        x = self.conv_out(fusion)
        x = self.split_and_stack(x)
        return x


class HDRPointwiseNN(nn.Module):
    def __init__(self, n_channel: int, net_input_size: int):
        self.n_channel = n_channel
        super().__init__()
        self.coeffs = Coeffs(
            channels=n_channel,
            nin=4, nout=3,
            luma_bins=8, channel_multiplier=1, spatial_bin=16, batch_norm=False, net_input_size=net_input_size
        )
        self.guide = GuideNN(n_channel)
        self.slice = Slice(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.apply_coeffs = ApplyCoeffs(n_channel)

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        return out
