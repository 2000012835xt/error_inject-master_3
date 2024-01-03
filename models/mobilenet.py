import torch.nn as nn


__all__ = ["mobilenetv1", "mobilenetv1_0p5"]


class MobileNetV1(nn.Module):
    def __init__(self, num_classes, chls):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, chls[0], 1),
            conv_dw(chls[0], chls[1], 1),
            conv_dw(chls[1], chls[2], 1),
            conv_dw(chls[2], chls[2], 1),
            conv_dw(chls[2], chls[3], 2),
            conv_dw(chls[3], chls[3], 1),
            conv_dw(chls[3], chls[4], 2),
            conv_dw(chls[4], chls[4], 1),
            conv_dw(chls[4], chls[4], 1),
            conv_dw(chls[4], chls[4], 1),
            conv_dw(chls[4], chls[4], 1),
            conv_dw(chls[4], chls[4], 1),
            conv_dw(chls[4], chls[5], 2),
            conv_dw(chls[5], chls[5], 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(chls[5], num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenetv1(pretrained=False, progress=True, device="cpu", **kwargs):
    return MobileNetV1(chls=[32, 64, 128, 256, 512, 1024], **kwargs)


def mobilenetv1_0p5(pretrained=False, progress=True, device="cpu", **kwargs):
    return MobileNetV1(chls=[16, 32, 64, 128, 256, 512], **kwargs)
