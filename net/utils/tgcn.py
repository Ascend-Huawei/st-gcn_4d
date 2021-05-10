# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        
        self.conv0 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=(t_kernel_size, 1),
                               padding=(t_padding, 0),
                               stride=(t_stride, 1),
                               dilation=(t_dilation, 1),
                               bias=bias)
        self.kc = out_channels * kernel_size
        self.c = out_channels
        self.v = 18
    
    def forward(self, x, A):

        x = self.conv(x)

        n, _, t, v = x.size()
     
        k = self.kernel_size
        kc = self.kc
        c = self.c
        ct = c*t
        x = x.view(n,k,ct,v) #View 'x' as 4D tensor (n,k,ct,v)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(n,c,t,k*v) # Before matmul, view 'x' as (n,c,t,k*v)
        
        a_k, a_v, a_w = A.size()        
        a_kv = a_k*a_v # Merge 3D tensor as 2D tensor before matmul with 'x'
        A = A.view(a_kv, a_w)
        
        x = torch.matmul(x, A) # Resultant -> (n,c,t,v)

        A = A.view(a_kv//a_v, a_v, a_w) #Unsquueze 'A' back to 3D tensor before return

        return x.contiguous(), A
