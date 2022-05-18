import argparse
import json
import numpy as np
import os
import os.path as osp
import torch

from torchviz import make_dot
from yolox.exp import get_exp
from yolox.utils import get_model_info
from gexporter import gexporter, codegen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOX DFG exporting")
    parser.add_argument("--exp_file", "-f", type=str, help="experiment description file")
    parser.add_argument("--cuda", action="store_true", help="running in cuda or not")
    parser.add_argument("--graph-name", '-gn', type=str, default="yolox_m.graph", help="graph saving name")
    parser.add_argument('--code-name', '-cn', type=str, default='yolox_m', help='source code saving name')
    parser.add_argument('--max-num-obj', '-mno', type=int, default=10000, help='maximum number of objects')
    parser.add_argument('--cutoff', '-cf', type=int, default=None, help='parameters cutoff for test')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file path')
    parser.add_argument('--size', type=int, nargs='+', default=[704, 1280], help='model input size')
    args = parser.parse_args()
    
    # construct model
    exp = get_exp(args.exp_file, None)
    model = exp.get_model()
    model.head.for_torchexe = True
    if args.cuda:
        model.cuda()
    model.eval()
    
    print(get_model_info(model, args.size))

    # load weight file
    if args.ckpt is not None:
        print('load checkpoint ...\n')
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    # execute forward pass
    h, w = args.size
    x = torch.rand(1, 3, h, w)
    if args.cuda:
        x = x.cuda()
    x = x.requires_grad_()
    y = tuple(model(x))
    
    # get data flow graph
    graph = gexporter(model, y)
    
    print('******** ops ********')
    ops = set()
    for v in graph.values():
        ops.add(v['name'])
    for o in ops: print(o)
    
    for k, v in graph.items():
        name = v['name'].split('Backward')[0]
        if 'Convolution' in name:
            name = 'Conv2d'
        elif 'BatchNorm' in name:
            name = 'BatchNorm2d'
        elif 'Silu' in name:
            name = 'SiLU'
        elif 'UpsampleNearest2D' in name:
            name = 'Upsample'
        elif 'MaxPool2DWithIndices' in name:
            name = 'MaxPool2d'
        graph[k]['name'] = name
    
    print('******** renamed ops ********')
    ops = set()
    for v in graph.values():
        ops.add(v['name'])
    for o in ops: print(o)
    
    # ThnnConvDepthwise2D
    for k, v in graph.items():
        if v['name'] != 'ThnnConvDepthwise2D':
            continue
        graph[k]['name'] = 'Conv2d'
        graph[k]['_saved_groups'] = graph[k]['_saved_self_sizes'][1]
    
    # add `_saved_shape` to view, for torchvision codebase
    for k, v in graph.items():
        if v['name'] != 'View':
            continue
        size = v['_saved_self_sizes']
        if len(size) == 4:
            graph[k]['_saved_shape'] = [size[0], 2, size[1] // 2, size[2], size[3]]
        elif len(size) == 5:
            graph[k]['_saved_shape'] = [size[0], -1, size[3], size[4]]
    
    # add `_saved_shape` to view, for Megvii codebase
    #for k, v in graph.items():
    #    if v['name'] != 'View':
    #        continue
    #    size = v['_saved_self_sizes']
    #    if len(size) == 4:
    #        graph[k]['_saved_shape'] = [size[0] * size[1] // 2, 2, size[2] * size[3]]
    #    elif len(size) == 3:
    #        graph[k]['_saved_shape'] = [size[0], -1, size[3], size[4]]
    
    # add index to the input blob name of Conv2d or Cat behind Split
    for k, v in graph.items():
        if v['name'] == 'Conv2d' and graph[v['input'][0]]['name'] == 'Split':
            v['input'][0] += '.1'
        elif v['name'] == 'Cat' and graph[v['input'][0]]['name'] == 'Split':
            v['input'][0] += '.0'
    
    print('******** renamed ops ********')
    ops = set()
    for v in graph.values():
        ops.add(v['name'])
    for o in ops: print(o)
    
    # build custom ops
    head_id = 0
    ops = []
    breaks = []
    # attrs is corresponding to this C struct:
    # struct alignas(float) Detection
    # {
    # 	float category;
    # 	struct
    # 	{
    # 		float x;
    # 		float y;
    # 		float width;
    # 		float height;
    # 	} box;
    # 	float score;
    # };
    attrs = ['category', 'x', 'y', 'width', 'height', 'score']
    down_sample_ratio = [8, 16, 32]
    for k, v in graph.items():
        if v['name'] != 'Output':
            continue
        breaks.append(v['input'][0]) # record insert positions
        op = dict(name='YOLOX', input=[v['input'][0]],
            max_num_obj=args.max_num_obj, down_sample_ratio=down_sample_ratio[head_id])
        ops.append([str(id(op)), op])
        graph[k]['input'][0] = str(id(op)) # update input op id
        batch = graph[k]['shape'][0]
        graph[k]['shape'] = [batch, args.max_num_obj * len(attrs) + 1, 1, 1] # the `+ 1` float buffer is used to save object number
        head_id = head_id + 1
    # insert custom ops
    new_graph = dict()
    head_id = 0
    for k, v in graph.items():
        new_graph[k] = v
        if head_id < len(breaks) and k == breaks[head_id]:
            new_graph[ops[head_id][0]] = ops[head_id][1]
            head_id = head_id + 1
    graph = new_graph
    
    # model source code generation
    cfd = osp.split(osp.realpath(__file__))[0] # current file directory
    code_dir = osp.join(cfd, osp.pardir, 'model', args.code_name)
    os.makedirs(code_dir, mode=0o777, exist_ok=True)
    path = osp.join(code_dir, args.code_name)
    codegen(model, graph, path, cutoff=args.cutoff)

    # DFG description file generation
    with open(args.graph_name, "w") as file:
        json.dump(graph, file, indent=2)
    
    # model visualization
    filename, _ = osp.splitext(args.graph_name)
    if not osp.exists(filename + '.pdf'):
        dot = make_dot(y, graph.keys(), params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
        dot.view(filename=filename)