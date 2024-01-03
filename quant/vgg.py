import torch
import torch.nn as nn

from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
from .quantizer import Quantizer
import torch.nn.functional as F
import pdb

class quant_ConvReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, weight, bias, scale0, scale1, scale2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.tensor(weight.cuda()))
        self.bias = nn.Parameter(torch.tensor(bias.cuda()))
        self.scale0 = nn.Parameter(torch.tensor(scale0))
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, all_positive=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        q_weight, _ = self.quan1(self.weight)
        y = F.conv2d(x, q_weight, self.bias, padding=self.padding)
        y = self.relu(y)
        y, yq = self.quan2(y)
        # y_integer = y / self.scale2
        return y, yq
    

class quant_LinearReLU(nn.Module):
    def __init__(self, in_channels, out_channels, weight, bias, scale1, scale2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shape = (out_channels, in_channels)
        # self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)
        # self.bias = nn.Parameter((torch.rand(self.out_channels)-0.5) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.tensor(weight.cuda()))
        self.bias = nn.Parameter(torch.tensor(bias.cuda()))
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, all_positive=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        q_weight, _ = self.quan1(self.weight)
        y = F.linear(x, q_weight, self.bias)
        y = self.relu(y)
        y, _ = self.quan2(y)
        return y
    

class quant_Linear(nn.Module):
    # 用在vgg最后一层，没有ReLU
    def __init__(self, in_channels, out_channels, weight, bias, scale1, scale2, zero_point):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shape = (out_channels, in_channels)
        # self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)
        # self.bias = nn.Parameter((torch.rand(self.out_channels)-0.5) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.tensor(weight.cuda()))
        self.bias = nn.Parameter(torch.tensor(bias.cuda()))
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, zero_point=zero_point, all_positive=False)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        q_weight, _ = self.quan1(self.weight)
        y = F.linear(x, q_weight, self.bias)
        y, _ = self.quan2(y)
        return y


stage = [64, 64, 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]


class quant_VGG16(nn.Module):
    def __init__(
        self, num_classes: int = 10, 
        input_scale=None, input_zero_point=None,
        conv_weights=None, conv_bias=None, 
        linear_weights=None, linear_bias=None,
        conv_in_scale=None,
        conv_w_scale=None, conv_a_scale=None, 
        linear_w_scale=None, linear_a_scale=None, zero_point=None,
        init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)

        self.input_scale = nn.Parameter(torch.tensor(input_scale))
        self.input_zero_point = nn.Parameter(torch.tensor(input_zero_point).float())
        self.input_quant = Quantizer(bit=8, scale=self.input_scale,
                zero_point=self.input_zero_point, all_positive=True)
        self.features = nn.ModuleList()

        i = 0
        in_channels = 3
        for v in stage:
            if v == "M":
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = cast(int, v) # 将v转换成整数
                QuantizedConvReLU2d = quant_ConvReLU2d(in_channels, v, conv_weights[i], conv_bias[i], conv_in_scale[i],
                                                       conv_w_scale[i], conv_a_scale[i], kernel_size=3, padding=1)
                self.features.append(QuantizedConvReLU2d)
                in_channels = v

                i += 1
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
            # nn.Linear(512, num_classes),
            quant_LinearReLU(512, 512, linear_weights[0], linear_bias[0], linear_w_scale[0], linear_a_scale[0]),
            quant_LinearReLU(512, 512, linear_weights[1], linear_bias[1], linear_w_scale[1], linear_a_scale[1]),
            quant_Linear(512, num_classes, linear_weights[2], linear_bias[2], linear_w_scale[2], linear_a_scale[2], zero_point)
        )
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.input_quant(x)
        y_integers=[]
        for idx, block in enumerate(self.features):
            if isinstance(block, quant_ConvReLU2d):
                x, y_integer = block(x)
                y_integers.append(y_integer)
            else:
                x=block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, y_integers

