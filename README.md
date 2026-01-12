
# QAT-Demo
Quantization Aware Training with pytorch

## quantization method
### dynamic quantization
> weight quantized ahead of time, but the activation/factor are dynamically quantized during inference. (hence dynamic)
### post-training static quantization
> convert the weights from float to int, as in dynamic quantization, but also performing fine-tune step of 
> feeding batches of data through the network and computer the resulting distributions of different activations.
> once fine-tune is complete, the weight and the input factor will be fixed(hence dynamic)
### quantization aware training
> all weights and activation are "face quantized" during both the forward and the backward passes of training. 
> Thus all the weight adjustments during training are made "aware" of the fact.(hence aware training)

## float train(contain qat with pytorch)
```commandline
python3 tools/train.py
```

## qat with pytorch-quantization
### ptq
```commandline
python3 tolls/train_ptq.py
```
### qat
```commandline
python3 tools/train_qat.py
```
### onnx-export and eval
```commandline
python3 tools/export.py
```

### engine-convert
```commandline
trtexec --onnx=${qat_onnx_path} --saveEngine=${engine_save_path} --int8
```
### engine-infer
```commandline

```

### analysis engine
```commandline
trt-engine-exploer
```

## Reference
• https://github.com/leimao/PyTorch-Quantization-Aware-Training
