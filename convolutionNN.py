import torch
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1
_input = torch.randn(batch_size, in_channels, width, height)
"""
padding:
     
"""
conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)
output = conv_layer(_input)
print(_input.shape)
print(output.shape)#1 10 98 98
print(conv_layer.weight.shape)#10 5 3 3
