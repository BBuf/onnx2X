## 模型转换命令

### 1. resnet50-v2-7.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\resnet50-v2-7.onnx --simplify_path .\models\resnet50-v2-7-simplify.onnx --pytorch_path .\models\resnet50-v2-7.pth --input_shape data:1,3,224,224

{'data': [1, 3, 224, 224]}
[-0.31065628  2.1637745   1.5490985  -0.3061202   1.1657124   1.8738675
  2.020359   -1.0334498  -0.16676627 -2.3562398 ]
[-0.31065452  2.1637743   1.5490973  -0.3061209   1.1657119   1.873866
  2.020358   -1.0334461  -0.16676697 -2.3562367 ]
mse 4.643172069052071e-12
```


### 2. mobilenetv2-7.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\mobilenetv2-7.onnx --simplify_path .\models\mobilenetv2-7-simplify.onnx --pytorch_path .\models\mobilenetv2-7.pth --input_shape input:1,3,224,224


{'input': [1, 3, 224, 224]}
[-1.9482288   0.28516793  1.1022013   0.44020158  1.8167695   3.3103495
  2.3945377  -0.3093502  -0.11157165 -1.749661  ]
[-1.9482299   0.28516483  1.1021987   0.44020024  1.8167661   3.3103478
  2.3945343  -0.30935067 -0.11157139 -1.7496638 ]
mse 3.6263321234741854e-12
```

## 3. bvlcalexnet-9.onnx


```sh
python .\onnx2pytorch.py --onnx_path .\models\bvlcalexnet-9.onnx --simplify_path .\models\bvlcalexnet-9-simplify.onnx --pytorch_path .\models\bvlcalexnet-9.pth --input_shape data_0:1,3,224,224


{'data_0': [1, 3, 224, 224]}
[0.00133673 0.00064852 0.00080947 0.00077721 0.00183716 0.0016319
 0.00229235 0.00044446 0.00085948 0.00061799]
[0.00133673 0.00064852 0.00080947 0.00077721 0.00183716 0.0016319
 0.00229235 0.00044446 0.00085948 0.00061799]
mse 4.7017316594428514e-20
```

### 4. googlenet-9.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\googlenet-9.onnx --simplify_path .\models\googlenet-9-simplify.onnx --pytorch_path .\models\googlenet-9.pth --input_shape data_0:1,3,224,224


{'data_0': [1, 3, 224, 224]}
[4.1070400e-04 5.9601903e-04 4.8920809e-04 1.3483929e-03 4.0508462e-03
 6.9703911e-03 1.5052327e-02 3.6285288e-05 3.9245504e-05 2.9938579e-05]
[4.1070409e-04 5.9601892e-04 4.8920832e-04 1.3483940e-03 4.0508453e-03
 6.9703949e-03 1.5052332e-02 3.6285299e-05 3.9245555e-05 2.9938616e-05]
mse 4.14498926989363e-17
```

## 5. squeezenet1.1-7.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\squeezenet1.1-7.onnx --simplify_path .\models\squeezenet1.1-7-simplify.onnx --pytorch_path .\models\squeezenet1.1-7.pth --input_shape data:1,3,224,224


{'data': [1, 3, 224, 224]}
[1.7816179  1.8798355  2.43737    4.018192   6.8265533  6.4349236
 6.1853323  4.2940426  3.1463337  0.79842544]
[1.7816175 1.8798358 2.4373698 4.0181923 6.8265524 6.4349227 6.1853323
 4.2940445 3.1463342 0.7984254]
mse 1.0111956827429935e-12
```

## 6. shufflenet-v2-10.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\shufflenet-v2-10.onnx --simplify_path .\models\shufflenet-v2-10-simplify.onnx --pytorch_path .\models\shufflenet-v2-10.pth --input_shape input:1,3,224,224


{'input': [1, 3, 224, 224]}
[ 1.4726675  -1.2103622   2.247375   -0.8061919   4.9881353  -0.82532334
  2.2503877  -2.571292   -3.546289   -2.362316  ]
[ 1.4726658  -1.2103636   2.2473738  -0.80619407  4.9881325  -0.82532525
  2.2503834  -2.5712945  -3.5462894  -2.36232   ]
mse 5.285994753023715e-12
```

## 7. inception-v1-9.onnx

```sh
python .\onnx2pytorch.py --onnx_path .\models\inception-v1-9.onnx --simplify_path .\models\inception-v1-9-simplify.onnx --pytorch_path .\models\inception-v1-9.pth --input_shape data_0:1,3,224,224



```
