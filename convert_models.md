## 模型转换命令

### 1. resnet50-v2-7.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\resnet50-v2-7.onnx --simplify_path .\models\resnet50-v2-7-simplify.onnx --pytorch_path .\models\resnet50-v2-7.pth --input_shape data:1,3,224,224

{'data': [1, 3, 224, 224]}
mse 4.643172069052071e-12
```


### 2. mobilenetv2-7.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\mobilenetv2-7.onnx --simplify_path .\models\mobilenetv2-7-simplify.onnx --pytorch_path .\models\mobilenetv2-7.pth --input_shape input:1,3,224,224


{'input': [1, 3, 224, 224]}
mse 3.6263321234741854e-12
```

## 3. bvlcalexnet-9.onnx


```sh
python .\onnx2pytorch.py --onnx_path .\models\bvlcalexnet-9.onnx --simplify_path .\models\bvlcalexnet-9-simplify.onnx --pytorch_path .\models\bvlcalexnet-9.pth --input_shape data_0:1,3,224,224


{'data_0': [1, 3, 224, 224]}
mse 4.7017316594428514e-20
```

### 4. googlenet-9.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\googlenet-9.onnx --simplify_path .\models\googlenet-9-simplify.onnx --pytorch_path .\models\googlenet-9.pth --input_shape data_0:1,3,224,224


{'data_0': [1, 3, 224, 224]}
mse 4.14498926989363e-17
```

## 5. squeezenet1.1-7.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\squeezenet1.1-7.onnx --simplify_path .\models\squeezenet1.1-7-simplify.onnx --pytorch_path .\models\squeezenet1.1-7.pth --input_shape data:1,3,224,224


{'data': [1, 3, 224, 224]}
mse 1.0111956827429935e-12
```

## 6. shufflenet-v2-10.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\shufflenet-v2-10.onnx --simplify_path .\models\shufflenet-v2-10-simplify.onnx --pytorch_path .\models\shufflenet-v2-10.pth --input_shape input:1,3,224,224


{'input': [1, 3, 224, 224]}
mse 5.285994753023715e-12
```

## 7. inception-v1-9.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\inception-v1-9.onnx --simplify_path .\models\inception-v1-9-simplify.onnx --pytorch_path .\models\inception-v1-9.pth --input_shape data_0:1,3,224,224


{'data_0': [1, 3, 224, 224]}
mse 1.6917238484094424e-17
```

## 8. inception-v2-9.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\inception-v2-9.onnx --simplify_path .\models\inception-v2-9-simplify.onnx --pytorch_path .\models\inception-v2-9.pth --input_shape data_0:1,3,224,224

{'data_0': [1, 3, 224, 224]}
mse 1.363866701867278e-15
```

## 9. mobilenetv2-1.0.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\mobilenetv2-1.0.onnx --simplify_path .\models\mobilenetv2-1.0-simplify.onnx --pytorch_path .\models\mobilenetv2-1.0.pth --input_shape data:1,3,224,224

{'data': [1, 3, 224, 224]}
mse 5.929283286576492e-12
```

## 10. vgg19-caffe2-9.onnx

- 可以看到模型转换成功了，但是Pytorch模型太大了，无法torch.save。

```sh
python .\onnx2pytorch.py --onnx_path .\models\vgg19-caffe2-9.onnx --simplify_path .\models\vgg19-caffe2-9-simplify.onnx --pytorch_path .\models\vgg19-caffe2-9.pth --input_shape data_0:1,3,224,224

{'data_0': [1, 3, 224, 224]}
mse 8.932564880980331e-19
Traceback (most recent call last):
  File ".\onnx2pytorch.py", line 92, in <module>
    convert_onnx_pytorch(model_slim, pytorch_model, output, input)
  File ".\onnx2pytorch.py", line 28, in convert_onnx_pytorch
    torch.save(model, pytorch_model)
  File "D:\Anaconda3\lib\site-packages\torch\serialization.py", line 372, in save
    _save(obj, opened_zipfile, pickle_module, pickle_protocol)
  File "D:\Anaconda3\lib\site-packages\torch\serialization.py", line 476, in _save
    pickler.dump(obj)
MemoryError
```

## 11. zfnet512-9.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\zfnet512-9.onnx --simplify_path .\models\zfnet512-9-simplify.onnx --pytorch_path .\models\zfnet512-9.pth --input_shape gpu_0/data_0:1,3,224,224

{'gpu_0/data_0': [1, 3, 224, 224]}
mse 5.180448423209617e-18
```