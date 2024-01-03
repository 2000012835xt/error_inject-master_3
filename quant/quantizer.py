import torch as t

class Quantizer(t.nn.Module):
    def __init__(self, bit, scale=1.0, zero_point=0, all_positive=False, symmetric=False, per_channel=False):
        super().__init__()

        self.bit = bit
        self.per_channel = per_channel
        self.scale = scale
        self.zero_point = zero_point


        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1


    # def init_from(self, x, *args, **kwargs):
    #     if self.per_channel:
    #         self.s = t.nn.Parameter(
    #             x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
    #     else:
    #         self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        # if self.per_channel:
        #     s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        # else:
        #     s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        # s_scale = grad_scale(self.s, s_grad_scale)

        x = x / self.scale
        xq = t.round(x) + self.zero_point
        xq = t.clamp(xq, self.thd_neg, self.thd_pos)
        # xq = t.round(x) + self.zero_point
        # x = (xq - self.zero_point) * self.scale 
        # print(self.zero_point, self.thd_neg, self.thd_pos)
        x = (xq - self.zero_point) * self.scale
        return x, xq
