import os
import io
import re
import onnx
import numpy as np
import torch
import onnxruntime as ort
import argparse
from tool import *

from onnx2pytorch import convert
from onnxsim import simplify

def convert_onnx_pytorch(onnx_model, pytorch_model, onnx_model_outputs, onnx_inputs):
    model = convert.ConvertModel(onnx_model, debug=False)
    model.eval()
    model.cpu()
    with torch.no_grad():
        outputs = model(onnx_inputs)
    if not isinstance(outputs, list):
        outputs = [outputs]
    outputs = [x.cpu().numpy() for x in outputs]
    # print(outputs[0][0][0:10])
    for output, onnx_model_output in zip(outputs, onnx_model_outputs):
        print("mse", ((onnx_model_output - output) ** 2).sum() / onnx_model_output.size)
        np.testing.assert_allclose(onnx_model_output, output, atol=1e-5, rtol=1e-3)
    
    torch.save(model, pytorch_model)

def get_onnx_output(onnx_model, onnx_inputs):
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    sess.set_providers(['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name : onnx_inputs.numpy()})
    # print(output[0][0][0:10])
    return output




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx2pytorch test")
    parser.add_argument("--onnx_path", default="", type=str, required=True)
    parser.add_argument("--simplify_path", default="", type=str, required=True)
    parser.add_argument("--pytorch_path", default="", type=str, required=True)
    parser.add_argument("--input_shape", default="input:1,3,224,224", type=str, required=True)
    args = parser.parse_args()
    
    input_shape_backup = args.input_shape
    input_shape = re.split(':', input_shape_backup)[-1]
    input_shape = re.split(',', input_shape)
    input = torch.randn(list(map(int, input_shape)))

    if(args.onnx_path.endswith('.onnx') == False):
        print('Please Check Your ONNX Model Path Format')
    if(args.simplify_path.endswith('.onnx') == False):
        print('Please Check Your ONNX Simplify Model Path Format')
    if(args.pytorch_path.endswith('.pth') == False):
        print('Please Check Your Pytorch Model Path Format')
    
    tool = Tool(args.onnx_path)
    for i, node in enumerate(tool.model.graph.node):
        if(node.op_type == "Dropout"):
            tool.remove_node(node)
    
    input_shape_backup = [input_shape_backup]

    input_shapes = {}

    if input_shape_backup is not None:
        for x in input_shape_backup:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes[name] = shape
    
    print(input_shapes)

    model_slim, check = simplify(tool.model, input_shapes=input_shapes)

    assert check, "Simplified ONNX model could not be validated"

    if args.simplify_path.endswith('.onnx'):
        onnx.save(model_slim, args.simplify_path)
    
    pytorch_model = args.pytorch_path
    output = get_onnx_output(model_slim, input)

    convert_onnx_pytorch(model_slim, pytorch_model, output, input)
    