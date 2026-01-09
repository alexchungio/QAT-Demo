import torch
import torch.nn as nn
import torch.nn.quantized
from torch.quantization import fuse_modules

from models.resnet import BasicBlock, Bottleneck, ResNet
from models.utils import _replace_relu

__all__ = ['qt_resnet50', 'qt_resnet101', 'qt_resnet152']


class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=False)
        # quantization compatibility
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.shortcut(x)
        else:
            identity = x
        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        """
        Fuses a list of modules into a single module for scale statistic
        :return:
        """
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample is not None:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        # quantization compatibility
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        out = self.add_relu.add_relu(out, identity)
        return out

    def fuse_model(self):
        """
        Fuses a list of modules into a single module for scale statistic
        :return:
        """
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2', 'relu2'],
                                               ['conv3', 'bn3']], inplace=True)
        if self.downsample is not None:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableResNet(ResNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableResNet, self).__init__(*args, **kwargs)
        # converts tensors from floating point to quantized, only used for input
        self.quant = torch.quantization.QuantStub()
        # convert tensor form quantized to floating point, only used for output
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.dequant(out)
        return out

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the models after modification is in floating point
        """

        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBasicBlock or type(m) == QuantizableBottleneck:
                m.fuse_model()


def _resnet(block, layers, quantize, **kwargs):
    model = QuantizableResNet(block, layers, **kwargs)
    _replace_relu(model)
    # if quantize:
    #     # TODO use pretrained as a string to specify the backend
    #     backend = 'fbgemm'

    return model


def qt_resnet18(num_classes=10, quantize=False):
    return _resnet(QuantizableBasicBlock, [2, 2, 2, 2], quantize=quantize, num_classes=num_classes)


def qt_resnet34(num_classes=10, quantize=False):
    return _resnet(QuantizableBasicBlock, [3, 4, 6, 3], quantize=quantize, num_classes=num_classes)


def qt_resnet50(num_classes=10, quantize=False):
    return _resnet(QuantizableBottleneck, [3, 4, 6, 3], quantize=quantize, num_classes=num_classes)


def qt_resnet101(num_classes=10, quantize=False):
    return _resnet(QuantizableBottleneck, [3, 4, 23, 3], quantize=quantize, num_classes=num_classes)


def qt_resnet152(num_classes=10, quantize=False):
    return _resnet(QuantizableBottleneck, [3, 8, 36, 3], quantize=quantize, num_classes=num_classes)


if __name__ == "__main__":
    import numpy as np

    dummy_input = torch.randn(10, 3, 32, 32, dtype=torch.float32)
    r50_q = qt_resnet50().eval()
    out_0 = r50_q(dummy_input)
    r50_q.fuse_model()
    out_1 = r50_q(dummy_input)
    assert np.allclose(out_0.detach().numpy(), out_1.detach().numpy(), rtol=1e-04, atol=1e-06)

    r50_q_int8 = qt_resnet50(num_classes=10, quantize=True)
    print(r50_q_int8)
    torch.onnx.export(r50_q_int8, (dummy_input,), 'r50_qt.onnx', input_names=['input'], output_names=['output'])