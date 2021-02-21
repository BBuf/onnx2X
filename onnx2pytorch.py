import os
import io

import onnx
import numpy as np
import torch
import onnxruntime as ort
import argparse

from onnx2pytorch import convert


def convert_onnx_pytorch(onnx_model, pytorch_model, onnx_model_outputs, onnx_inputs):
    model = convert.ConvertModel(onnx_model)
    model.eval()
    model.cpu()
    
    with torch.no_grad():
        outputs = model(onnx_inputs)
        print(outputs[0][:10])

    torch.save(model, pytorch_model)

    if not isinstance(outputs, list):
        outputs = [outputs]

    outputs = [x.cpu().numpy() for x in outputs]

    for output, onnx_model_output in zip(outputs, onnx_model_outputs):
        print("mse", ((onnx_model_output - output) ** 2).sum() / onnx_model_output.size)
        np.testing.assert_allclose(onnx_model_output, output, atol=1e-5, rtol=1e-3)

def get_onnx_output(onnx_model, onnx_inputs):
    input_shape = torch.Tensor(1, 2)
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    sess.set_providers(['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    output = sess.run([output_name], {input_name : onnx_inputs.numpy()})
    print(output[0:10])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx2pytorch test")
    parser.add_argument("--onnx_path", default="", type=str, required=True)
    parser.add_argument("--pytorch_path", default="", type=str, required=True)
    args = parser.parse_args()

    input = torch.randn(1, 3, 640, 640)
    
    onnx_model = onnx.load(args.onnx_path)
    pytorch_model = args.pytorch_path
    output = get_onnx_output(onnx_model, input)
    convert_onnx_pytorch(onnx_model, pytorch_model, output, input)
