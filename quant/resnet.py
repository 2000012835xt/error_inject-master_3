import torch
import torch.nn as nn
import os
from .quantizer import Quantizer
import torch.nn.functional as F
import pdb

# __all__ = [
#     "ResNet",
#     "resnet18",
#     "resnet34",
#     "resnet50",
# ]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class quant_ConvReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, weight, bias, 
                 scale_in, scale_w, scale_out, zero_point_out,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = weight.cuda()
        self.bias = bias.cuda()
        self.scale_in = scale_in
        self.scale_w = scale_w
        self.scale_out = scale_out
        self.zero_point_out = zero_point_out

        self.quan1 = Quantizer(bit=8, scale=scale_w, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale_out, all_positive=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        q_weight = self.quan1(self.weight)
        y = F.conv2d(x, q_weight, self.bias, stride=self.stride, padding=self.padding)
        y = self.relu(y)
        y = self.quan2(y)

        return y
    

class quant_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, weight, bias, 
                 scale_in, scale_w, scale_out, zero_point_out,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = weight.cuda()
        self.bias = bias.cuda()
        self.scale_in = scale_in
        self.scale_w = scale_w
        self.scale_out = scale_out
        self.zero_point_out = zero_point_out

        self.quan1 = Quantizer(bit=8, scale=scale_w, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale_out, zero_point=zero_point_out, all_positive=False)

    def forward(self, x):
        q_weight = self.quan1(self.weight)
        y = F.conv2d(x, q_weight, self.bias, stride=self.stride, padding=self.padding)
        y = self.quan2(y)

        return y
    

class quant_Linear(nn.Module):
    # 用在最后一层，没有ReLU
    def __init__(self, in_channels, out_channels, weight, bias, scale1, scale2, zero_point):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shape = (out_channels, in_channels)
        # self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)
        # self.bias = nn.Parameter((torch.rand(self.out_channels)-0.5) * 0.001, requires_grad=True)
        self.weight = weight.cuda()
        self.bias = bias.cuda()
        self.scale1 = scale1
        self.scale2 = scale2
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, zero_point=zero_point, all_positive=False)

    def forward(self, x):
        q_weight = self.quan1(self.weight)
        y = F.linear(x, q_weight, self.bias)
        y = self.quan2(y)

        return y
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        conv_weights=None, conv_bias=None,
        conv_input_scale=None, 
        conv_w_scale=None, 
        conv_out_scale=None, 
        conv_out_zero_point=None,
        add_scale=None,
        add_zero_point=None

    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1 = quant_ConvReLU2d(inplanes, planes,conv_weights[0], conv_bias[0],
                                     conv_input_scale[0], conv_w_scale[0], 
                                    conv_out_scale[0], conv_out_zero_point[0], stride=stride)
        
        self.conv2 = quant_Conv2d(planes, planes, conv_weights[1], conv_bias[1],
                                     conv_input_scale[1], conv_w_scale[1], 
                                    conv_out_scale[1], conv_out_zero_point[1])
        
        self.add_quant = Quantizer(bit=8, scale=add_scale, zero_point=add_zero_point, all_positive=True)

    def forward(self, x):
        identity = x
        # pdb.set_trace()
        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.add_quant(out)
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34_quant(nn.Module):
    def __init__(
        self, num_classes=1000,
        conv_weights=None, conv_bias=None, conv_w_scale=None,
        conv_input_scale=None, conv_input_zero_point = None,
        conv_out_scale = None, conv_out_zero_point = None,
        add_scale=None, add_zero_point=None,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None
    ):
        super(ResNet34_quant, self).__init__()

        self.block = BasicBlock
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        # self.conv1 = nn.Conv2d(
        #     3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        # )
        # # END

        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)

        self.input_quant = Quantizer(bit=8, scale=conv_input_scale[0], zero_point=conv_input_zero_point[0], all_positive=False)

        self.conv1 = quant_ConvReLU2d(3, self.inplanes, conv_weights[0], conv_bias[0], conv_input_scale[0], 
                                      conv_w_scale[0], conv_out_scale[0], conv_out_zero_point[0],
                                      kernel_size=7, stride=2, padding=3)
              
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0], conv_weights[1:7], conv_bias[1:7], conv_input_scale[1:7], 
                                      conv_w_scale[1:7], conv_out_scale[1:7], conv_out_zero_point[1:7],
                                      add_scale[0:3], add_zero_point[0:3])
        
        self.layer2 = self._make_layer(
            block, 128, layers[1], conv_weights[7:16], conv_bias[7:16], 
            conv_input_scale[7:16], conv_w_scale[7:16], conv_out_scale[7:16], conv_out_zero_point[7:16],
            add_scale[3:7], add_zero_point[3:7],
            stride=2, dilate=replace_stride_with_dilation[0]
        )

        self.layer3 = self._make_layer(
            block, 256, layers[2], conv_weights[16:29], conv_bias[16:29], 
            conv_input_scale[16:29], conv_w_scale[16:29], conv_out_scale[16:29], conv_out_zero_point[16:29],
            add_scale[7:13], add_zero_point[7:13],
            stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], conv_weights[29:36], conv_bias[29:36], 
            conv_input_scale[29:36], conv_w_scale[29:36], conv_out_scale[29:36], conv_out_zero_point[29:36],
            add_scale[13:], add_zero_point[13:],
            stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = quant_Linear(512 * block.expansion, num_classes, conv_weights[36], 
                               conv_bias[36], conv_w_scale[36], conv_out_scale[36],
                               conv_out_zero_point[36])
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, conv_weights, conv_bias, 
                    conv_input_scale, conv_w_scale, 
                    conv_out_scale, conv_out_zero_point, 
                    add_scale, add_zero_point,
                    stride=1, dilate=False):
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )
            downsample = quant_Conv2d(self.inplanes, planes * block.expansion, conv_weights[2], conv_bias[2],
                                      conv_input_scale[2], conv_w_scale[2], conv_out_scale[2], 
                                      conv_out_zero_point[2], kernel_size=1, stride=stride, padding=0)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                conv_weights[0:2], conv_bias[0:2],
                conv_input_scale[0:2], conv_w_scale[0:2], 
                conv_out_scale[0:2], conv_out_zero_point[0:2],
                add_scale[0], add_zero_point[0]
            )
        )

        self.inplanes = planes * block.expansion

        if stride == 1:
            # 除去list中的前2个元素构成新的list
            new_conv_weights = conv_weights[2:]
            new_conv_bias = conv_bias[2:]
            new_conv_input_scale = conv_input_scale[2:]
            new_conv_w_scale = conv_w_scale[2:]
            new_conv_out_scale = conv_out_scale[2:]
            new_conv_out_zero_point = conv_out_zero_point[2:]

        else:
            # 除去list中的前3个元素构成新的list
            new_conv_weights = conv_weights[3:]
            new_conv_bias = conv_bias[3:]
            new_conv_input_scale = conv_input_scale[3:]
            new_conv_w_scale = conv_w_scale[3:]
            new_conv_out_scale = conv_out_scale[3:]
            new_conv_out_zero_point = conv_out_zero_point[3:]


        new_add_scale = add_scale[1:]
        new_add_zero_point = add_zero_point[1:]

        # pdb.set_trace()
        for i in range(0, blocks-1):
            
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    conv_weights=[new_conv_weights[2*i],new_conv_weights[2*i+1]], 
                    conv_bias=[new_conv_bias[2*i],new_conv_bias[2*i+1]],
                    conv_input_scale=[new_conv_input_scale[2*i],new_conv_input_scale[2*i+1]], 
                    conv_w_scale=[new_conv_w_scale[2*i],new_conv_w_scale[2*i+1]], 
                    conv_out_scale=[new_conv_out_scale[2*i],new_conv_out_scale[2*i+1]], 
                    conv_out_zero_point=[new_conv_out_zero_point[2*i],new_conv_out_zero_point[2*i+1]],
                    add_scale=new_add_scale[i],
                    add_zero_point=new_add_zero_point[i]
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # pdb.set_trace()
        x = self.input_quant(x)
        x = self.conv1(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


# def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         script_dir = os.path.dirname(__file__)
#         state_dict = torch.load(
#             script_dir + "/pretrained/" + arch + ".pt", map_location=device
#         )
#         model.load_state_dict(state_dict)
#     return model


# def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet(
#         "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
#     )


# def resnet34(pretrained=False, progress=True, device="cpu", **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet(
#         "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, **kwargs
#     )


# def resnet50(pretrained=False, progress=True, device="cpu", **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet(
#         "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, **kwargs
#     )
