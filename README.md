# ONNX2NN

ONNX作为微软的网络模型中间表示被各个框架广泛应用，包括Pytroch，TensorFlow，OneFlow，Keras，Paddle等多种深度学习训练框架。因此，一直在思考一个问题，一个TensorFlow导出来的ONNX模型是否可以借助ONNX被Pytorch框架使用呢？ONNX的理想是作为所有框架的模型的中间交换，那么我们只需要再实现ONNX到各个框架的逆转就可以完成这件事情了。本工程的目的即是尝试支持ONNX转换到各种训练框架，主要为了锻炼算子对齐和更深入的了解ONNX。

# 代码结构

```markdown
- onnx2pytorch onnx转pytorch代码实现
- onnx2pytorch.py onnx转pytorch测试代码
- convert_models.md 转换ONNX Model Zoo里面的模型对应的命令和结果记录
- README.md 
```

# 运行环境

- pytorch >= 1.1.0
- onnx>=1.8.1
-   >=1.6.0
- onnxoptimizer>=0.2.3

# 使用方法

使用下面的命令将各个训练框架导出的ONNX模型转换成Pytorch模型

```sh
python .\onnx2pytorch.py ...
```

参数列表如下:

- `--onnx_path` 字符串，必选参数，代表onnx模型的路径
- `--pytorch_path` 字符串，必选参数，代表转换出的Pytorch模型保存路径
- `--simplify_path` 字符串，可选参数，代表ONNX模型简化（例如删除Dropout和常量OP）后保存的ONNX模型路径
- `--input_shape` 字符串，必选参数，代表ONNX模型的输入数据层的名字和维度信息

# 使用示例

```sh
python .\onnx2pytorch.py --onnx_path .\models\mobilenetv2-7.onnx --simplify_path .\models\mobilenetv2-7-simplify.onnx --pytorch_path .\models\mobilenetv2-7.pth --input_shape input:1,3,224,224
```

# 模型转换失败处理方法

- 将`onnx2pytorch.py`里面的`model = convert.ConvertModel(onnx_model, debug=False)`这行代码里面的`debug`设置False重新运行模型即可定位到转换失败的OP，然后你可以在工程提出issue或者自己解决然后给本工程PR。

# ONNX2Pytorch

## 已支持的ONNX OP

- [x] Conv
- [x] BatchNormalization
- [x] GlobalAvgragePool
- [x] AvgPool
- [x] MaxPool
- [x] BatchNorm
- [x] Flatten
- [x] Reshape
- [x] Relu
- [x] Add
- [x] Gemm
- [x] Sigmoid
- [x] Mul
- [x] Concat
- [x] Resize (还有一些问题需要解决，当前版本支持固定倍数方法)
- [x] Transpose
- [x] LRN
- [x] Clip
- [x] Pad2d
- [x] Split
- [x] ReduceMean
- [x] LeakyRelu

## 已验证支持的模型

基于ONNXRuntime和Pytorch推理之后特征值mse小于1e-7，视为转换成功

### 分类模型
- [x] zfnet512-9.onnx
- [x] resnet50-v2-7.onnx
- [x] mobilenetv2-7.onnx
- [x] mobilenetv2-1.0.onnx
- [x] bvlcalexnet-9.onnx
- [x] googlenet-9.onnx
- [x] squeezenet1.1-7.onnx
- [x] shufflenet-v2-10.onnx
- [x] inception-v1-9.onnx
- [x] inception-v2-9.onnx
- [x] vgg19-caffe2-9.onnx
- [x] rcnn-ilsvrc13-9.onnx

### 检测模型
- [x] yolov5s-simple.onnx
 
### 分割模型

# TODO

- [ ] 支持更多模型
- [ ] 重构工程，并解决某些模型转为Pytorch模型之后Netron可视化看不到某些OP的问题
- [ ] 一些部署工作，比如Keras导出的ONNX转为Pytorch模型后，二次导出ONNX递交给NCNN推理

# 相关链接

- https://github.com/ToriML/onnx2pytorch
- https://github.com/daquexian/onnx-simplifier