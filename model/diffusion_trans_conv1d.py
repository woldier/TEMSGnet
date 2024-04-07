"""
Diffusion model unet 1d implementation
"""
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from inspect import isfunction  # Determining whether a variable is a function
from einops.layers.torch import Rearrange  # Tensor reshape tool
from einops import rearrange, reduce
from functools import partial


def exists(x):
    """
     Determine whether the input x is None, if it is None, return False, otherwise return True.
    :param x:需要判断的值
    :return: True/False
    """
    return x is not None


def default(val, d):
    """
     a default function, if val is None, then replace it with the default value d.
    :param val: variable or function
    :param d:  Default value or default function
    :return:
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    """
    Convert the number into an array, the number of each group is determined by the divisor
    such as num_to_groups(12,4) output is [4,4,4].
    :param num: 总数
    :param divisor: 分组中的数目
    :return:
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# ==============================Residual==========================
class Residual(nn.Module):
    """
     Residual block , here fc can be any realization,
     but it should be noted that the output of fc must have the same dimension as the input.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# ==============================Upsample==========================
def Upsample(dim, dim_out=None):
    """
    Unet Upper Sampling Module
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),  # (h,w) -> (2h,2w)
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),  # (H,W) -> (H-(3-1)+2, W-(3-1)+2) -> (H,W)
    )


# ==============================Downsample==========================
def Downsample(dim, dim_out=None):
    """

    :param dim:
    :param dim_out:
    :return:
    """
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1)  -> b (c p1 ) h ", p1=2),  # (b,c,h) -> (b,2*c,h/2)
        # the convolution kernel is 1. After the convolution,
        # the width and height remain the same, but the number
        # of channels changes to dim or dim_out, depending on whether dim_out is None.
        nn.Conv1d(dim * 2, default(dim_out, dim), 1)
        # If dim_out = None then (b,2*c,h/2,w/2) -> (b,c,h/2,w/2) where c is equivalent to dim
        # If dim_out is not None then (b,2*c,h/2,w/2) -> (b,dim_out,h/2,w/2)
    )


# ========================time-embedding====================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Since the noise parameters about time (noise level) are shared by the network,
    we use sinusoidal position embeddings here to encode time t (inspired by Transformer's position encoder).
    This allows the network to know that we are currently at that particular time step t (noise level).

    The input t-shape of this module is [batch_size,1]
    (for a batch of data sampling time t may be different),
    and the input shape after this king is [batch_size,dim].
    Here dim is the parameter given when we initialize
    the model, which is the dimension of the embedding.

    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.squeeze()


#  ==========================WeightStandardizedConv1d==========================
class WeightStandardizedConv1d(nn.Conv1d):
    """
     The ddpm paper uses the Wide ResNet block,
     an Unet implementation in which the network replaces
     the original base 2d convolutional layer with a convolutional layer called weight normalization.
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        """
        Inherited from nn.Conv2d
        Correct his normalized weights on top of this one
        :param x:
        :return:
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        # The dimensionality reduction specifies that the strategy is mean, which is equivalent to doing GroupNormal.
        mean = reduce(weight, "o ... -> o 1 1 ", "mean")
        # Here the rule of reduce is to call a function through which
        # to get the final value, here is the calculation of variance
        var = reduce(weight, "o ... -> o 1 1 ",
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# ================================Unet block========================================
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        # (b,dim,h,w) -> (b,dim_out,h+2-3+1,w+2-3+1) -> ((b,dim_out,h,w))
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
        # GroupNorm
        self.norm = nn.GroupNorm(groups, dim_out)
        # SiLU激活
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            #  shift is the equivalent of performing a numerical transformation
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# =================Resnet块============================
class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),  # SiLU
                nn.Linear(time_emb_dim, dim_out * 2)  # mlp
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out,
                            groups=groups)  # Convolutional layer, hw constant Number of channels dim ->dim_out
        self.block2 = Block(dim_out, dim_out, groups=groups)  # The number of channels and hw are unchanged.
        # If the input dim is not equal to the final output dim
        # we do a convolutional mapping, if it is equal
        # we do a fully-connected mapping, and then use this as his residual block.
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None  # transformation scale
        # If mlp is not None, and the incoming time_emb exists,
        # we mlp the time_emb so that the image can be added to our time_embed.
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)  # (b, time_emb) -> (b,out_dim * 2)
            time_emb = rearrange(time_emb, "b c -> b c 1 ")  # (b, out_dim * 2) -> (b,out_dim * 2, 1)
            # Split the input tensor in a given dimension (axis), in this case into 2 parts in dimension 1.
            scale_shift = time_emb.chunk(2, dim=1)
        # Pass in the previously split time_emb (b,out_dim, 1, 1) ,(b,out_dim, 1, 1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)  # Convolution results are summed with residuals


# ================================Attention======================================
class Attention(nn.Module):
    """

    Traditional Multihead Self-Attention Mechanism

    The input and output shapes after this block are the same
    """

    def __init__(self, dim, heads=32, dim_head=64):
        """

        :param dim:
        :param heads:
        :param dim_head:
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h_w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  #
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)  #
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)  # softmax

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h h_w d -> b (h d) h_w", h_w=h_w)  # reshape
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    线性attention
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h_w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c h_w -> b (h c) h_w", h=self.heads, h_w=h_w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# ======================Unet主体===================================
class Unet1D(nn.Module):
    """
    Unet network for predicting ep(xt,t)
    Input xt [batch_size,channels,height,width] and time t[batch_size,1] for this network
    Output res [batch_size,channels,height,width]

    1. first, a convolutional layer is added to the batch of noisy images to positionally embed the noise layer
    2. Next, a series of downsampling stages are applied. Each downsampling
        stage consists of 2 ResNet blocks + GroupNormal + Attention + Residential + down sample operation. 3.
    3. In the middle of the network, the ResNet blocks are applied again, interleaved with Attention.
    4. Next, a series of up-sampling stages are applied. Each up-sampling stage consists
        of 2 ResNet blocks + GroupNormal + Attention + Residential + up sample operation.
    5. The final output of the network is obtained by adding a convolutional block to a Residential block.


    """

    def __init__(
            self,
            dim=32,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            self_condition=False,
            resnet_block_groups=4,
    ):
        """

        :param dim:
        :param init_dim: Initialize the number of channels,
            corresponding to the number of channels of the up and down
            sampling base, if not given, then the same as dim(img_size)
        :param out_dim: Output channels
        :param dim_mults: Downsampling channel coefficients Determines the number of
            up and down sampling channels together with init_dim [inti_dim, inti_dim*dim_mults]
        :param channels: input channel
        :param self_condition: Is it a conditional diffusion model
        :param resnet_block_groups:
        """
        super().__init__()

        # determine dimensions
        self.channels = channels  # Number of image channels
        self.self_condition = self_condition  # Is it a conditional diffusion model
        # In the case of a conditional diffusion model the number of input channels is twice that of the original siganl
        input_channels = channels * (2 if self_condition else 1)
        # If initialization dim is given Measurement use int_dim Otherwise use default dim for signal dim
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1, padding=0)  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]  # Number of channels for downsampling
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)  # Initialize ResnetBlock Set groups parameter

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(  #
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])  #
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        # A residual block with convolution, time_embedding, two convolutions,
                        # and finally, residuals, where the input channel equals the output.
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        #  A residual block with convolution, time_embedding,
                        #  two convolutions, and finally, residuals, where the input channel equals the output.
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        # Residuals block Attention and GroupNorm before residuals
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        # Downsampling Number of channels from dim_in->dim_out
                        Downsample(dim_in, dim_out)
                        # If it is the last downsampling, do a convolution dim_in -> dim_out
                        if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        # A residual block with convolution, time_embedding, two convolutions,
        # and finally, residuals, where the input channel equals the output.
        self.mid_block1 = block_klass(mid_dim, mid_dim,
                                      time_emb_dim=time_dim)
        # Residuals block Attention and GroupNorm before residuals
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        # A residual block with convolution, time_embedding, two convolutions,
        # and finally, residuals, where the input channel equals the output.
        self.mid_block2 = block_klass(mid_dim, mid_dim,
                                      time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        #  If out_dim is not empty, it is the same as the number of input channels
        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:  # If the conditional diffusion model
            # If the given x_self_cond is None,
            # it means that we currently want to train unconditional to set the condition input to all zeros.
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            # Splicing in dimension 1 doubles the number of channels.

        x = self.init_conv(x)

        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)  # 拼接
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class TEMSGnet(nn.Module):
    def __init__(self, dim=32, channels=1, gama=0.5):
        super().__init__()

        self.u1 = Unet1D(dim=dim, channels=channels, self_condition=True)
        self.gama = gama

    def forward(self, x, t, condition):
        out = self.u1(x, t, condition)
        return out
