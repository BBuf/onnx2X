#coding=utf-8
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper

class Tool(object):
    # 初始化onnx模型
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)
        self.model = onnx.shape_inference.infer_shapes(self.model)
        self.inputs = []
        self.outputs = []

    # 保存onnx模型
    def export(self, save_path):
        onnx.checker.check_model(self.model)
        self.model = onnx.shape_inference.infer_shapes(self.model)
        onnx.save(self.model, save_path)
    
    # 获取onnx模型的输入，返回一个列表
    def get_input_names(self):
        set_input = set()
        set_initializer = set()
        for ipt in self.model.graph.input:
            set_input.add(ipt.name)
        for x in model.graph.initializer:
            set_initializer.add(x.name)
        return list(set_input - set_initializer)
    
    # 为onnx模型增加batch维度
    def set_model_input_batch(self, index=0, name=None, batch_size=4):
        model_input = None
        if name is not None:
            for ipt in self.model.graph.input:
                if ipt.name == name:
                    model_input = ipt
        else:
            model_input = self.model.graph.input[index]
        if model_input:
            tensor_dim = model_input.type.tensor_type.shape.dim
            tensor_dim[0].ClearField("dim_param")
            tensor_dim[0].dim_value = batch_size
        else:
            print('get model input failed, check index or name')
        
    # 为onnx模型的输入设置形状
    def set_model_input_shape(self, index=0, name=None, shape=None):
        model_input = None
        if name is not None:
            for ipt in self.model.graph.input:
                if ipt.name == name:
                    model_input = ipt
        else:
            model_input = self.model.graph.input[index]
        if model_input:
            if shape is not None:
                tensor_shape_proto = model_input.type.tensor_type.shape
                tensor_shape_proto.ClearField("dim")
                tensor_shape_proto.dim.extend([])
                for d in shape:
                    dim = tensor_shape_proto.dim.add()
                    dim.dim_value = d
            else:
                print('get input shape failed, check input')
        else:
            print('get model input failed, check index or name')
    
    # 通过名字获取onnx模型中的计算节点
    def get_node_by_name(self, name):
        for node in self.model.graph.node:
            if node.name == name:
                return node
    
    # 通过op的类型获取onnx模型的计算节点
    def get_nodes_by_optype(self, typename):
        nodes = []
        for node in self.model.graph.node:
            if node.op_type == typename:
                nodes.append(node)
        return nodes

    # 通过名字获取onnx模型计算节点的权重
    def get_weight_by_name(self, name):
        for weight in self.model.graph.initializer:
            if weight.name == name:
                return weight
    
    # 设置权重，注意这个weight是TensorProto类型，`https://github.com/onnx/onnx/blob/b1e0bc9a31eaefc2a9946182fbad939843534984/onnx/onnx.proto#L461`
    def set_weight(self, weight, data_numpy=None, all_ones=False, all_zeros=False):
        if data_numpy is not None:
            raw_shape = tuple([i for i in weight.dims])
            new_shape = np.shape(data_numpy)
            if weight.data_type == 8:
                print("Can NOT handle string data type right now...")
                exit()
            if new_shape != raw_shape:
                print("Warning: the new weight shape is not consistent with original shape!")
                weight.dims[:] = list(new_shape)
                for model_input in self.model.graph.input:
                    if model_input.name == weight.name:
                        # copy from onnx.helper...
                        tensor_shape_proto = model_input.type.tensor_type.shape
                        tensor_shape_proto.ClearField("dim")
                        tensor_shape_proto.dim.extend([])
                        for d in new_shape:
                            dim = tensor_shape_proto.dim.add()
                            dim.dim_value = d

            weight.ClearField("float_data")
            weight.ClearField("int32_data")
            weight.ClearField("int64_data")
            weight.raw_data = data_numpy.tobytes()
        else:
            if all_ones:
                wr = numpy_helper.to_array(weight)
                wn = np.ones_like(wr)
            elif all_zeros:
                wr = numpy_helper.to_array(weight)
                wn = np.zeros_like(wr)
            else:
                print("You must give a data_numpy to set the weight, or set the all_ones/all_zeros flag.")
                exit()
            weight.ClearField("float_data")
            weight.ClearField("int32_data")
            weight.ClearField("int64_data")
            weight.raw_data = wn.tobytes()

    # 通过名字设置ONNX节点的权重
    def set_weight_by_name(self, name, data_numpy=None, all_ones=False, all_zeros=False):
        weight = self.get_weight_by_name(name)
        self.set_weight(weight, data_numpy, all_ones, all_zeros)
    
    # 移除ONNX模型中的目标节点
    def remove_node(self, target_node):
        '''
            删除只有一个输入和输出的节点
        '''
        node_input = target_node.input[0]
        node_output = target_node.output[0]
        # 将后继节点的输入设置为目标节点的前置节点
        for node in self.model.graph.node:
            for i, n in enumerate(node.input):
                if n == node_output:
                    node.input[i] = node_input

        target_names = set(target_node.input) & set([weight.name for weight in self.model.graph.initializer])
        self.remove_weights(target_names)
        target_names.add(node_output)
        self.remove_inputs(target_names)
        self.remove_value_infos(target_names)
        self.model.graph.node.remove(target_node)

    # 移除ONNX模型中指定节点的权重
    def remove_weights(self, name_list):
        rm_list = []
        for weight in self.model.graph.initializer:
            if weight.name in name_list:
                rm_list.append(weight)
        for weight in rm_list:
            self.model.graph.initializer.remove(weight)

    # 移除ONNX模型中指定的输入节点
    def remove_inputs(self, name_list):
        rm_list = []
        for input_t in self.model.graph.input:
            if input_t.name in name_list:
                rm_list.append(input_t)
        for input_t in rm_list:
            self.model.graph.input.remove(input_t)

    # 移除ONNX模型中指定的输入输出节点
    def remove_value_infos(self, name_list):
        rm_list = []
        for value_info in self.model.graph.value_info:
            if value_info.name in name_list:
                rm_list.append(value_info)
        for value_info in rm_list:
            self.model.graph.value_info.remove(value_info)
    
    # 给ONNX模型中的目标节点设置指定属性
    def set_node_attribute(self, target_node, attr_name, attr_value):
        flag = False
        for attr in target_node.attribute:
            if (attr.name == attr_name):
                if attr.type == 1:
                    attr.f = attr_value
                elif attr.type == 2:
                    attr.i = attr_value
                elif attr.type == 3:
                    attr.s = attr_value
                elif attr.type == 4:
                    attr.t = attr_value
                elif attr.type == 5:
                    attr.g = attr_value
                # NOTE: For repeated composite types, we should use something like
                # del attr.xxx[:]
                # attr.xxx.extend([n1, n2, n3])
                elif attr.type == 6:
                    attr.floats[:] = attr_value
                elif attr.type == 7:
                    attr.ints[:] = attr_value
                elif attr.type == 8:
                    attr.strings[:] = attr_value
                else:
                    print("unsupported attribute data type with attribute name")
                    return False
                flag = True

        if not flag:
            # attribute not in original node
            print("Warning: you are appending a new attribute to the node!")
            target_node.attribute.append(helper.make_attribute(attr_name, attr_value))
            flag = True
        return flag

    def chunk_at(self, target_node):
        r_nodes = [target_node]
        r_input_names = [input_n for input_n in target_node.input]
        r_count = len(r_nodes) + len(r_input_names)

        while True:
            for node in self.model.graph.node:
                # print("nn", node.output)
                if node in r_nodes:
                    continue
                for o in node.output:
                    if o in r_input_names:
                        r_nodes.append(node)
                        r_input_names.extend([input_n for input_n in node.input])
                        continue
            n_count = len(r_nodes) + len(r_input_names)
            if n_count == r_count:
                break
            r_count = n_count

        print("debug r count", r_count)

        d_nodes = []
        d_inputs = []
        d_weights = []
        d_value_infos = []
        for node in self.model.graph.node:
            if node not in r_nodes:
                d_nodes.append(node)
        for model_input in self.model.graph.input:
            if model_input.name not in r_input_names:
                d_inputs.append(model_input)
        for weight in self.model.graph.initializer:
            if weight.name not in r_input_names:
                d_weights.append(weight)
        for value_info in self.model.graph.value_info:
            if value_info.name not in r_input_names:
                d_values.append(value_info)
        for node in d_nodes:
            self.model.graph.node.remove(node)
        for model_input in d_inputs:
            self.model.graph.input.remove(model_input)
        for weight in d_weights:
            self.model.graph.initializer.remove(weight)
        for value_info in d_value_infos:
            self.model.graph.value_info.remove(value_info)

        target_node.output[0] = self.model.graph.output[0].name
        # remove other outputs if model has multi-output
        d_outputs = []
        for i, output in enumerate(self.model.graph.output):
            if i != 0 :
                d_outputs.append(output)
        for output in d_outputs:
            self.model.graph.output.remove(output)

    # 在指定节点前插入flatten node
    def insert_flatten_before(self, target_node):
        # get target_node inputs
        node_input = target_node.input[0]
        # create new node
        node_name = "flatten_test"
        flatten_node = helper.make_node('Flatten', inputs=[node_input], outputs=[node_name], name=node_name)
        # set target_node inputs to new node outputs
        target_node.input[0] = node_name
        for target_node_index, _target_node in enumerate(self.model.graph.node):
            if _target_node == target_node:
                self.model.graph.node.insert(target_node_index, flatten_node)
                break

    # 在指定节点target_node前插入一个新的OP
    def insert_op_before(self, node_name, target_node, input_idx=0, *args, **kwargs):
        '''
        op_name
        weight_dict
        attr_dict
        ......
        NOTE:
        you must ensure the output shape match the input shape of target_node
        '''
        # get target_node inputs
        node_input = target_node.input[input_idx]
        weight_input = []
        weight_input_vi = []
        weight_initializer = []
        if "weight_dict" in kwargs:
            for weight_name, weight_numpy in kwargs["weight_dict"].items():
                weight_input.append(weight_name)
                weight_input_vi.append(
                        helper.make_tensor_value_info(
                            name=weight_name,
                            elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_numpy.dtype],
                            shape=weight_numpy.shape
                        )
                )
                weight_initializer.append(
                    helper.make_tensor(
                            name=weight_name,
                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_numpy.dtype],
                            dims=weight_numpy.shape,
                            vals=weight_numpy.tobytes(),
                            raw=True
                    )
                )
        # create new node
        new_op_node = helper.make_node(
                                kwargs["op_name"],
                                inputs=[node_input, *weight_input],
                                outputs=[node_name],
                                name=node_name,
                                **kwargs["attr_dict"]
                            )
        # set target_node input to new node outputs
        target_node.input[input_idx] = node_name
        # TODO: change other nodes input into the new node?
        # iterator all the nodes in the graph and find
        # which node's input equals the original target_node input
        # ...
        # add new node and weight input into the graph
        for target_node_index, _target_node in enumerate(self.model.graph.node):
            if _target_node == target_node:
                self.model.graph.node.insert(target_node_index, new_op_node)
                break
        self.model.graph.input.extend(weight_input_vi)
        self.model.graph.initializer.extend(weight_initializer)

    # 将target_node添加到ONNX模型中作为输出节点
    def add_extra_output(self, target_node, output_name):
        target_output = target_node.output[0]
        extra_shape = []
        for vi in self.model.graph.value_info:
            if vi.name == target_output:
                extra_elem_type = vi.type.tensor_type.elem_type
                for s in vi.type.tensor_type.shape.dim:
                    extra_shape.append(s.dim_value)
        extra_output = helper.make_tensor_value_info(
                                output_name,
                                extra_elem_type,
                                extra_shape
                            )
        identity_node = helper.make_node('Identity', inputs=[target_output], outputs=[output_name], name=output_name)
        self.model.graph.node.append(identity_node)
        self.model.graph.output.append(extra_output)