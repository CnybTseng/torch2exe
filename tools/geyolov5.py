import argparse
import json
import numpy as np
import os
import os.path as osp
import sys
import torch

sys.path.append(osp.join(osp.split(osp.realpath(__file__))[0], osp.pardir, osp.pardir, 'yolov5'))

from gexporter import gexporter, codegen
from models.experimental import attempt_load
from torchviz import make_dot
from utils.torch_utils import select_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolov5 graph exporting')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--ckpt', type=str, help='checkpoint file path')
    parser.add_argument('--size', type=int, nargs='+', default=[704, 1280], help='model input size')
    parser.add_argument("--graph-name", '-gn', type=str, default="yolov5m6.graph", help="graph saving name")
    parser.add_argument('--code-name', '-cn', type=str, default='yolov5m6', help='source code saving name')
    parser.add_argument('--max-num-obj', '-mno', type=int, default=1000, help='maximum number of objects')
    parser.add_argument('--cutoff', '-cf', type=int, default=None, help='parameters cutoff for test')
    args = parser.parse_args()

    device = select_device(args.device)
    model = attempt_load(args.ckpt, map_location=device, inplace=False, fuse=False)
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad_()
    
    for k, v in dict(model.state_dict()).items():
        print(f'{k:<50} {v.shape}')
    
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
    anchor_grid = anchor_grid.detach().cpu().numpy()
    print(f'anchor_grid: {anchor_grid.shape}')

    h, w = args.size
    x = torch.zeros(1, 3, h, w).to(device).type_as(next(model.parameters())).requires_grad_()
    y = tuple(model(x))
    # model.33.anchors will make warning
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
    down_sample_ratio = [8, 16, 32, 64]
    for k, v in graph.items():
        if v['name'] != 'Output':
            continue
        breaks.append(v['input'][0]) # record insert positions
        op = dict(name='YOLOV5', input=[v['input'][0]],
            anchor=anchor_grid[head_id].reshape(-1).astype(np.int).tolist(),
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