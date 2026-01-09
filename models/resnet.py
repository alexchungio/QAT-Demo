import torch
import torch.nn as nn
from .utils import fuse_conv_and_bn
import pytorch_quantization.nn as quant_nn

__all__ = ['BasicBlock', 'Bottleneck', 'ResNet', 'resnet50', 'resnet101', 'resnet152']


class FuseModel(nn.Module):
    def __int__(self):
        super(FuseModel, self).__int__()

    def fuse_conv_bn(self):

        for n, m in self.named_modules():
            need_to_fuse = []
            conv_bn = [None, None]
            for sub_n, sub_m in m.named_children():
                if isinstance(sub_m, (nn.Conv2d, quant_nn.QuantConv2d)):
                    conv_bn[0] = sub_n
                elif isinstance(sub_m, (nn.BatchNorm2d)):
                    conv_bn[1] = sub_n

                if None not in conv_bn:
                    # fused_cb = fuse_conv_and_bn(m.get_submodule(conv_bn[0]), m.get_submodule(conv_bn[1]))
                    fused_cb = fuse_conv_and_bn(m.__getattr__(conv_bn[0]), m.__getattr__(conv_bn[1]))
                    need_to_fuse.append([conv_bn[0], conv_bn[1], fused_cb])
                    conv_bn = [None, None]

            for i in need_to_fuse:
                m.__setattr__(i[0], i[2])
                m.__delattr__(i[1])

    def fuse_model(self):
        self.fuse_conv_bn()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes=None, planes=None, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        out += identity
        out = self.relu(out)
        return out


class ResNet(FuseModel):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16
        # this layer is different from torchvision.resnet18() since this models adopted for Cifar100
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if stride != 1 or self.in_planes != block.expansion * planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, block.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.expansion * planes)
                )
            else:
                downsample = None
            layers.append(block(self.in_planes, planes, stride, downsample))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def fuse_model(self):
        pass


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == "__main__":
    from torchstat import stat
    from thop import profile

    dummy_input = torch.rand(1, 3, 32, 32)
    model = resnet50(num_classes=10)
    out = model(dummy_input)
    # stat(models, (3, 32, 32))
    flops, params = profile(model, (dummy_input, ), verbose=False)
    print(f"params(MB):{params * 1e-6: .6f}, MACs(GB):{flops * 1e-9: .6f}")
