import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexConv2d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None, bias=False):
        super().__init__()
        if p is None:
            p = (k[0] // 2, k[1] // 2)
        # Construct complex convolution to preserve phase information
        self.real = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=bias)
        self.imag = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=bias)

    def forward(self, x):
        # The real part and the virtual part are divided from the channel dimension
        xr, xi = torch.chunk(x, 2, dim=1)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None, o=(0, 0), bias=False):
        super().__init__()
        if p is None:
            p = (k[0] // 2, k[1] // 2)
        self.real = nn.ConvTranspose2d(
            cin, cout, k, stride=s, padding=p, output_padding=o, bias=bias
        )
        self.imag = nn.ConvTranspose2d(
            cin, cout, k, stride=s, padding=p, output_padding=o, bias=bias
        )

    def forward(self, x):
        xr, xi = torch.chunk(x, 2, dim=1)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexBN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(c)
        self.bn_i = nn.BatchNorm2d(c)

    def forward(self, x):
        xr, xi = torch.chunk(x, 2, dim=1)
        xr = self.bn_r(xr)
        xi = self.bn_i(xi)
        return torch.cat([xr, xi], dim=1)


class ComplexPReLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.act_r = nn.PReLU(c)
        self.act_i = nn.PReLU(c)

    def forward(self, x):
        # Activate layers are applied to the real and imaginary parts respectively
        xr, xi = torch.chunk(x, 2, dim=1)
        return torch.cat([self.act_r(xr), self.act_i(xi)], dim=1)


class CConvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None):
        super().__init__()
        self.conv = ComplexConv2d(cin, cout, k, s, p, bias=False)
        self.bn = ComplexBN(cout)
        self.act = ComplexPReLU(cout)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CDeconvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), o=(0, 0), p=None):
        super().__init__()
        self.deconv = ComplexConvTranspose2d(cin, cout, k, s, p, o, bias=False)
        self.bn = ComplexBN(cout)
        self.act = ComplexPReLU(cout)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))



class DPRNNBlock(nn.Module):
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, bidir_intra=True):
        super().__init__()
        # intra-GRU
        self.intra = nn.GRU(
            Cg, hidden_intra, num_layers=1, batch_first=True, bidirectional=bidir_intra
        )
        intra_out = hidden_intra * (2 if bidir_intra else 1)
        self.intra_proj = nn.Linear(intra_out, Cg)
        self.ln_intra = nn.LayerNorm(Cg)
        # inter-GRU
        self.inter = nn.GRU(
            Cg, hidden_inter, num_layers=1, batch_first=True, bidirectional=False
        )
        self.inter_proj = nn.Linear(hidden_inter, Cg)
        self.ln_inter = nn.LayerNorm(Cg)

    def forward(self, xg):
        B, Cg, T, Freq = xg.shape

        # intra-GRU over freq dimension
        h = (
            xg.permute(0, 2, 3, 1)
            .contiguous()
            .view(B * T, Freq, Cg)
        )
        h, _ = self.intra(h)
        h = self.intra_proj(h)
        h = self.ln_intra(h)
        h = (
            h.view(B, T, Freq, Cg)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        xg = xg + h
        # inter-GRU over time dimension
        z = (
            xg.permute(0, 3, 2, 1)
            .contiguous()
            .view(B * Freq, T, Cg)
        )
        z, _ = self.inter(z)
        z = self.inter_proj(z)
        z = self.ln_inter(z)
        z = (
            z.view(B, Freq, T, Cg)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        xg = xg + z
        return xg


class DPRNNBottleneck(nn.Module):
    # Stack multiple DPRNNBlocks
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, num_blocks=2, bidir_intra=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DPRNNBlock(Cg, hidden_intra, hidden_inter, bidir_intra)
                for _ in range(num_blocks)
            ]
        )
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x



class SinusoidalPE1D(nn.Module):
    # Sinusoidal Positional Encoding for 1D sequences
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: (d_model // 2)])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0).to(x.device, x.dtype)


class FramewiseMHSA(nn.Module):
    def __init__(self, Cg, d_model=256, nhead=4, dropout=0.0):
        super().__init__()
        self.pre = nn.Linear(Cg, d_model, bias=False)
        self.pe = SinusoidalPE1D(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.post = nn.Linear(d_model, Cg, bias=False)
        # learnable gate for residual fusion
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):  
        B, Cg, T, Freq = x.shape
        # pool over frequency to get frame-level tokens
        g = x.mean(dim=-1).permute(0, 2, 1).contiguous()  
        h = self.pre(g)  
        h = self.pe(h)
        # multi-head self-attention
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)
        ff = self.ffn(h)
        h = self.norm2(h + ff)
        # project back and expand to original shape
        h = self.post(h).permute(0, 2, 1).unsqueeze(-1)  
        h = h.expand(-1, -1, -1, Freq)
        return x + torch.sigmoid(self.alpha) * h


class DCCRN_DPRNN(nn.Module):
    def __init__(
        self,
        c_in=2,
        base=32,
        hid_intra=64,
        hid_inter=64,
        num_dp_blocks=2,
        bidir_intra=True,
        fba_d_model=128,
        fba_heads=4,
        fba_dropout=0.0,
    ):
        super().__init__()

        self.in_proj = nn.Conv2d(c_in, 2, kernel_size=1, bias=False)

        c1, c2, c3 = base, base * 2, base * 2
        self.e1 = CConvBlock(cin=1, cout=c1, k=(3, 3), s=(1, 1))
        self.e2 = CConvBlock(cin=c1, cout=c2, k=(3, 3), s=(1, 2))  
        self.e3 = CConvBlock(cin=c2, cout=c3, k=(3, 3), s=(1, 2))  

        Cg = 2 * c3
        self.dp = DPRNNBottleneck(
            Cg=Cg,
            hidden_intra=hid_intra,
            hidden_inter=hid_inter,
            num_blocks=num_dp_blocks,
            bidir_intra=bidir_intra,
        )
        # frame-level full-band attention
        self.fba = FramewiseMHSA(
            Cg=Cg, d_model=fba_d_model, nhead=fba_heads, dropout=fba_dropout
        )
        
        self.d3 = CDeconvBlock(cin=c3, cout=c2, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d2 = CDeconvBlock(cin=c2 * 2, cout=c1, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d1 = CConvBlock(cin=c1 * 2, cout=c1, k=(3, 3), s=(1, 1))

        self.out_r = nn.Conv2d(c1, 1, kernel_size=1)
        self.out_i = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _cat_align(a, b):
        F = min(a.size(-1), b.size(-1))
        return torch.cat([a[..., :F], b[..., :F]], dim=1)

    def forward(self, x):
        e1 = self.e1(x)  
        e2 = self.e2(e1)  
        e3 = self.e3(e2)  

        yb = self.dp(e3) 
        yb = self.fba(yb)

        d3 = self.d3(yb) 
        d3 = self._cat_align(d3, e2)
        d2 = self.d2(d3)  
        d2 = self._cat_align(d2, e1)
        d1 = self.d1(d2)  

        dr, di = torch.chunk(d1, 2, dim=1)
        mr = torch.tanh(self.out_r(dr))
        mi = torch.tanh(self.out_i(di))
        return torch.cat([mr, mi], dim=1)



