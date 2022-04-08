import torch
from torch import nn

# we only need torch.nn.modules.KEY_WORD.x
KEY_WORDS = [
'linear',
'conv',
'activation',
'pooling',
'normalization',
'batchnorm',
'instancenorm',
'padding',
'upsampling',
'flatten',
'channelshuffle',
]

if __name__ == '__main__':
    ops = nn.modules.__all__
    with open('ops.txt', 'w') as file:
        for op in ops:
            module = nn.modules.__dict__.get(op)
            fullname = str(module).split('\'')[1]
            key_word = fullname.split('.')[3]
            is_ops = False
            if key_word in KEY_WORDS:
                is_ops = True
            file.write(f'{op:>30}   {fullname:<60}   {is_ops}\n')