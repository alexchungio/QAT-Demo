import torch
from torch import nn
import torch.backends.quantized
import pytorch_quantization.nn as quant_nn

__all__ = [
    "fuse_conv_and_bn",
]

def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def fuse_conv_and_bn(conv, bn):
    # Fuse quantized convolution and batchnorm layers for NVIDIA pytorch_quantization

    # Extract underlying conv if it's a quantized module
    if hasattr(conv, 'conv'):
        # This is a quantized conv layer
        base_conv = conv.conv
    else:
        # This is a regular conv layer
        base_conv = conv

    quant_fusedconv = quant_nn.QuantConv2d(
        in_channels=base_conv.in_channels,
        out_channels=base_conv.out_channels,
        kernel_size=base_conv.kernel_size,
        stride=base_conv.stride,
        padding=base_conv.padding,
        groups=base_conv.groups,
        bias=True
    ).requires_grad_(True).to(base_conv.weight.device)

    # fusion weight of conv and bn
    weight_conv = base_conv.weight.clone().view(base_conv.out_channels, -1)
    weight_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    quant_fusedconv.weight.data.copy_(torch.mm(weight_bn, weight_conv).view(quant_fusedconv.weight.shape))

    # fusion bias of conv and bn
    bias_conv = torch.zeros(base_conv.weight.size(0),
                         device=base_conv.weight.device) if base_conv.bias is None else base_conv.bias
    bias_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    quant_fusedconv.bias.data.copy_(torch.mm(weight_bn, bias_conv.reshape(-1, 1)).reshape(-1) + bias_bn)

    # Preserve quantization parameters
    if hasattr(conv, '_input_quantizer'):
        quant_fusedconv._input_quantizer = conv._input_quantizer
    if hasattr(conv, '_weight_quantizer'):
        quant_fusedconv._weight_quantizer = conv._weight_quantizer

    return quant_fusedconv
