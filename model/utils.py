import torch
import torch.backends.quantized


def quantize_model(model, backend, input_size=(1, 3, 32, 32)):
    _dummy_input_data = torch.rand(**input_size)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError('Quantized backend not supported')
    torch.backends.quantized.engine = backend
    model.eval()
    # make sure that weight qconfig match that of the serialized model
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)
    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

    return
