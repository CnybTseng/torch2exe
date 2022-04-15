import json
import torch

from torch import nn
from torchviz import make_dot
from gexporter import gexporter, codegen

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 32, 3, bias=False)
    def forward(self, input):
        output = self.act1(self.bn1(self.conv1(input)))
        return self.conv2(output)

if __name__ == '__main__':
    model = Model().cuda()
    model.eval()
    blob = torch.rand(1, 3, 512, 512).cuda()
    blob = blob.requires_grad_()
    output = model(blob)
    print(output.shape)
    graph = gexporter(model, tuple(output))
    codegen(model, graph, 'testres/test', cutoff=None)
    with open('test.graph', "w") as file:
        json.dump(graph, file, indent=2)
    for k, v in dict(model.named_parameters()).items():
        print(f'{k} {v.shape}')
    dot = make_dot(tuple(output), graph.keys(), params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
    dot.view(filename='test')