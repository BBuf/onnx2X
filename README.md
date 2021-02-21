# ONNX2NN

ONNX作为微软的网络模型中间表示被各个框架广泛应用，包括Pytroch，TensorFlow，OneFlow，Keras，Paddle等多种深度学习训练框架。因此，一直在思考一个问题，一个TensorFlow导出来的ONNX模型是否可以借助ONNX被Pytorch框架使用呢？ONNX的理想是作为所有框架的模型的中间交换，那么我们只需要再实现ONNX到各个框架的逆转就可以完成这件事情了。本工程的目的即是尝试支持ONNX转换到各种训练框架，主要为了锻炼算子对齐和更深入的了解ONNX。

# 代码结构

```markdown
- onnx2pytorch onnx转pytorch代码实现
- onnx2pytorch.py onnx转pytorch测试代码
- download_models.sh 下载ONNX官方提供的各种模型脚本
- README.md 
```

# ONNX2Pytorch

## 已支持的ONNX OP

- [x] Conv
- [x] BatchNormalization
- [x] GlobalAveragePool
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

## 已验证支持的模型

基于ONNXRuntime和Pytorch推理之后特征值差距小于1e-7，视为转换成功

- [x] ResNet18
- [x] MobileNetV2
- [x] YOLOV5-s

# TODO

- [ ] TensorFlow导出的ONNX转为指定框架（{Pytorch/OneFlow）
- [ ] Keras导出的ONNX转为指定框架（Pytorch/OneFlow）
- [ ] 结合onnx-simplifier简化模型
- [ ] 自动生成转化后的模型代码
- [ ] 一些部署工作，比如Keras导出的ONNX转为Pytorch模型后，二次导出ONNX递交给NCNN推理

