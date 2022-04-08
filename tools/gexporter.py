import json
import numpy as np
import os
import struct
import torch
import warnings

from collections import defaultdict

def gexporter(model, output):
    '''export data flow graph of the model.
    '''
    parameters = dict(model.named_parameters())
    param_map = {id(v): k for k, v in parameters.items()}
    modules = dict(model.named_modules())
    graph = defaultdict(dict)
    seen = set()
    ags = set() # AccumulateGrads next to model input tensor
    ANP = 1 # AccumulateGrad next to parameter tensor
    ANI = 2 # AccumulateGrad next to model input tensor
    SAVED_PREFIX = "_saved_"
    is_cuda = next(model.parameters()).is_cuda # model in GPU
    cbws = list() # ConvolutionBackward corresponding to Conv2d with bias
    vbws = list() # ViewBackwards corresponding to Conv2d with bias
    abws = list() # AddBackward0s corresponding to Conv2d with bias

    def size_to_str(size):
        return '[' + (', ').join(['%d' % v for v in size]) + ']'

    def add_node(fn, parent):
        assert not torch.is_tensor(fn) # non leaf node
        if fn in seen and fn not in ags:
            return
        
        seen.add(fn)
        if hasattr(fn, 'variable'): # it's a AccumulateGrad next to leaf node
            var = fn.variable
            seen.add(var)
            # add leaf node to the parameters of parent (ConvolutionBackward/BatchNormBackward/ViewBackward/...)
            pname = param_map.get(id(var), None)
            if pname is not None: # non model input tensor, but a parameter tensor
                wname = pname.split('.')[-1]
                assert wname in ['weight', 'bias']
                mname = pname[:len(pname) - len('.' + wname)]
                module = modules[mname]
                # parameter for Conv2d
                if isinstance(module, torch.nn.Conv2d) and is_cuda:
                    if wname == 'weight':
                        if module.bias is not None:
                            # unlike BatchNorm2d, bias don't connectd to ConvolutionBackward through AccumulateGrad
                            graph[str(id(parent))]['bias'] = (mname + '.bias')#.replace('.', '_')
                            cbws.append(str(id(parent)))
                    else: # ViewBackward next to AddBackward0
                        vbws.append(str(id(parent)))
                graph[str(id(parent))][wname] = pname#.replace('.', '_')
                return ANP # no need to track backward any more
            else: # model input tensor
                ags.add(fn)
                # add model input to graph
                graph[str(id(var))] = dict(name='Input', shape=list(var.shape))
                return ANI # no need to track backward any more

        # add fn, i.e operator to graph
        operator = dict(name=str(type(fn).__name__), input=list())
        
        if hasattr(fn, 'next_functions'):
            children = list()
            for nfn in fn.next_functions:
                if nfn[0] is not None:
                    ret = add_node(nfn[0], fn)
                    # update the input list of this fn
                    if ret != ANP and ret != ANI:
                        operator['input'].append(str(id(nfn[0])))
                    elif ret == ANI:
                        var = nfn[0].variable
                        operator['input'].append(str(id(var)))
                    children.append(str(id(nfn[0])))
            for child in children:
                if child in cbws: # AddBackward0 next to ConvolutionBackward
                    abws.append(str(id(fn)))
        
        # copy attributes from fn
        for attr in dir(fn):
            if not attr.startswith(SAVED_PREFIX):
                continue
            try:
                val = getattr(fn, attr)
            except Exception as e:
                print(f'getattr failed: {e}')
                continue
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                operator[attr + '_sizes'] = list(val.shape)
            elif isinstance(val, torch.Size) and val.numel() > 0:
                operator[attr] = list(val)
            else:
                operator[attr] = val
        
        for k, v in operator.items():
            graph[str(id(fn))][k] = v
    
    def add_base_tensor(var): # model output tensor
        if var in seen:
            return
        seen.add(var)
        if var.grad_fn:
            add_node(var.grad_fn, var)
            graph[str(id(var))] = dict(name='Output', input=[str(id(var.grad_fn))], shape=list(var.shape))
    
    if isinstance(output, tuple):
        for var in output:
            add_base_tensor(var)
    else:
        add_base_tensor(output)

    # delete ViewBackwards, only necessary on CUDA mode
    for vb in vbws:
        del graph[vb]
    
    # delete AddBackward0s, only necessary on CUDA mode
    for ab in abws:
        inputs = graph[ab]['input']
        if len(inputs) != 2:
            continue
        # ID of the input ConvolutionBackward for this AddBackward0
        conv_id = inputs[0] if inputs[0] in cbws else inputs[1]
        for k, v in graph.items():
            if 'input' not in v.keys(): # exclude Input
                continue
            for i, in_id in enumerate(v['input']):
                if in_id == ab:
                    graph[k]['input'][i] = conv_id
        del graph[ab]
    
    return graph

def codegen(model, graph, path, namespace='algorithm', cutoff=None):
    '''model code generation.
    '''
    _, _name = os.path.split(path)
    name, _ = os.path.splitext(_name)
    
    # generate header file
    header = path + '.h'
    with open(header, 'wb') as file:
        file.write(f'#ifndef {name.upper()}_H_\n'.encode())
        file.write(f'#define {name.upper()}_H_\n\n'.encode())
        file.write('#include "algorithm.h"\n\n'.encode())
        file.write(f'namespace {namespace} {{\n\n'.encode())
        file.write(f'extern "C" ALGORITHM_API const char graph[];\n'.encode())
        file.write(f'extern "C" ALGORITHM_API const size_t graph_len;\n'.encode())
        file.write(f'extern "C" ALGORITHM_API const char *id[];\n'.encode())
        file.write(f'extern "C" ALGORITHM_API const size_t id_count;\n'.encode())
        file.write(f'extern "C" ALGORITHM_API const float weight[];\n'.encode())
        file.write(f'extern "C" ALGORITHM_API const size_t weight_count;\n\n'.encode())
        file.write(f'}} // namespace {namespace}\n\n'.encode())
        file.write(f'#endif // {name.upper()}_H_\n'.encode())

    # generate source file
    source = path + '.cpp'
    with open(source, 'wb') as file:
        # include and namespace
        file.write(f'#include "{name}.h"\n\n'.encode())
        file.write(f'namespace {namespace} {{\n\n'.encode())

        # json format data flow graph
        strs = json.dumps(graph)
        file.write('const char graph[] = {\n'.encode())       
        carr = [f'\'{s}\'' for s in strs]
        carr = (',').join(carr)
        file.write(f'{carr}\n}};\n\n'.encode())
        file.write('const size_t graph_len = sizeof(graph);\n\n'.encode())

        # id array
        file.write('const char *id[] = '.encode())
        keys = [f'\"{k}\"' for k in graph.keys()]
        strs = '{\n' + (',\n').join(keys) + '\n};'
        file.write(f'{strs}\n\n'.encode())
        file.write('const size_t id_count = sizeof(id) / sizeof(char *);\n\n'.encode())

        # weight array
        file.write('const float weight[] = '.encode())
        parameters = dict(model.state_dict())
        param_array = []
        for k, v in graph.items():
            if v.get('weight') is not None:
                weight = parameters[v['weight']]
                param_array += [weight.detach().cpu().reshape(-1).numpy()]
            if v.get('bias') is not None:
                bias = parameters[v['bias']]
                param_array += [bias.detach().cpu().reshape(-1).numpy()]
            if v.get('name') is not None and v['name'] == 'BatchNorm2d':
                rmkey = v['weight'].replace('weight', 'running_mean')
                running_mean = parameters[rmkey]
                param_array += [running_mean.detach().cpu().reshape(-1).numpy()]
                rvkey = v['weight'].replace('weight', 'running_var')
                running_var = parameters[rvkey]
                param_array += [running_var.detach().cpu().reshape(-1).numpy()]
        param_array = np.concatenate(param_array)
        vals = [float(e).hex() for e in param_array[:cutoff]]
        strs = '{\n' + (',').join(vals) + '\n};'
        file.write(f'{strs}\n\n'.encode())
        file.write('const size_t weight_count = sizeof(weight) / sizeof(float);\n\n'.encode())
        
        # namespace
        file.write(f'}} // namespace {namespace}\n\n'.encode())
        
        # simple validation
        total_state_dict = sum([v.numel() for k, v in parameters.items() if 'num_batches_tracked' not in k])
        total_write = len(vals)
        if total_state_dict != total_write:
            warnings.warn(f'{total_state_dict} != {total_write}')