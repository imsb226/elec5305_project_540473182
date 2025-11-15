import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================== Complex basic modules =====================

class ComplexConv2d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None, bias=False):
        """
        Complex 2D convolution.
        Convention:
          - Input channels = 2 * cin  (real, imag)
          - Output channels = 2 * cout
        """
        super().__init__()
        if p is None:
            p = (k[0] // 2, k[1] // 2)
        # Real and imaginary convolutions share the same input size but have independent weights.
        self.real = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=bias)
        self.imag = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=bias)

    def forward(self, x):
        # Split real and imaginary parts along the channel dimension.
        xr, xi = torch.chunk(x, 2, dim=1)  
        # Standard complex convolution: (W_r + j W_i) * (x_r + j x_i)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        # Re-concatenate back into complex representation.
        return torch.cat([yr, yi], dim=1)  


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None, o=(0, 0), bias=False):
        """
        Complex 2D transposed convolution (deconvolution).
        Convention:
          - Input channels = 2 * cin
          - Output channels = 2 * cout
        """
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
        """
        Complex batch norm implemented as two independent real-valued BN layers:
          - Input channels = 2 * c
          - BN is applied separately to real and imag parts.
        """
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
        """
        Complex PReLU with separate learnable parameters for real and imaginary parts:
          - Input channels = 2 * c
        """
        super().__init__()
        self.act_r = nn.PReLU(c)
        self.act_i = nn.PReLU(c)

    def forward(self, x):
        xr, xi = torch.chunk(x, 2, dim=1)
        return torch.cat([self.act_r(xr), self.act_i(xi)], dim=1)


class CConvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None):
        """
        Complex conv block: ComplexConv2d + ComplexBN + ComplexPReLU.
        cin, cout are complex channel counts (per real/imag part).
        Effective input channels = 2 * cin, output = 2 * cout.
        """
        super().__init__()
        self.conv = ComplexConv2d(cin, cout, k, s, p, bias=False)
        self.bn = ComplexBN(cout)
        self.act = ComplexPReLU(cout)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CDeconvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), o=(0, 0), p=None):
        """
        Complex deconv block: ComplexConvTranspose2d + ComplexBN + ComplexPReLU.
        cin, cout are complex channel counts.
        """
        super().__init__()
        self.deconv = ComplexConvTranspose2d(cin, cout, k, s, p, o, bias=False)
        self.bn = ComplexBN(cout)
        self.act = ComplexPReLU(cout)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


# ===================== DPRNN & MHSA bottleneck =====================

class DPRNNBlock(nn.Module):
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, bidir_intra=True):
        super().__init__()
        # Intra-chunk GRU over frequency axis.
        self.intra = nn.GRU(
            Cg, hidden_intra, num_layers=1, batch_first=True, bidirectional=bidir_intra
        )
        intra_out = hidden_intra * (2 if bidir_intra else 1)
        self.intra_proj = nn.Linear(intra_out, Cg)
        self.ln_intra = nn.LayerNorm(Cg)

        # Inter-chunk GRU over time axis.
        self.inter = nn.GRU(
            Cg, hidden_inter, num_layers=1, batch_first=True, bidirectional=False
        )
        self.inter_proj = nn.Linear(hidden_inter, Cg)
        self.ln_inter = nn.LayerNorm(Cg)

    def forward(self, xg):
        """
        xg: [B, Cg, T, Freq]
        """
        B, Cg, T, Freq = xg.shape

        # Intra GRU: operate along frequency, for each time frame independently.
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
        xg = xg + h  # residual fusion

        # Inter GRU: operate along time, for each frequency bin independently.
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
        xg = xg + z  # residual fusion
        return xg


class DPRNNBottleneck(nn.Module):
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, num_blocks=2, bidir_intra=True):
        """
        Stack of DPRNN blocks forming the bottleneck.
        """
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
    def __init__(self, d_model: int, max_len: int = 4096):
        """
        Standard sinusoidal positional encoding for 1D sequences.
        """
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
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0).to(x.device, x.dtype)


class FramewiseMHSA(nn.Module):
    def __init__(self, Cg, d_model=256, nhead=4, dropout=0.0):
        """
        Frame-wise multi-head self-attention over time.
        Input Cg is the real-valued bottleneck channel dimension.
        """
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
        # Learnable gate controlling how much attention output is fused back.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        x: [B, Cg, T, Freq]
        """
        B, Cg, T, Freq = x.shape

        # Average over frequency to get one token per time frame.
        g = x.mean(dim=-1).permute(0, 2, 1).contiguous() 

        # Project to transformer dimension and add positional encoding.
        h = self.pre(g)
        h = self.pe(h)

        # Self-attention over time.
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)

        # Feed-forward network with residual connection.
        ff = self.ffn(h)
        h = self.norm2(h + ff)

        # Project back to Cg and broadcast over frequency bins.
        h = self.post(h).permute(0, 2, 1).unsqueeze(-1)   
        h = h.expand(-1, -1, -1, Freq)                  

        # Gated residual fusion.
        return x + torch.sigmoid(self.alpha) * h


# ===================== Main network: DCCRN_DPRNN =====================

class DCCRN_DPRNN(nn.Module):
    def __init__(
        self,
        c_in=2,          # Input channels, typically 2: [real, imag]
        base=32,
        hid_intra=96,
        hid_inter=96,
        num_dp_blocks=2,
        bidir_intra=True,
        fba_d_model=128,
        fba_heads=4,
        fba_dropout=0.0,
    ):
        super().__init__()

        # Optional projection if input has more than 2 channels.
        # Currently not used in forward; kept here for future extension.
        self.in_proj = nn.Conv2d(c_in, 2, kernel_size=1, bias=False)

        c1, c2, c3 = base, base * 2, base * 2

        # Encoder: cin/cout are complex channel counts.
        self.e1 = CConvBlock(cin=1,  cout=c1, k=(3, 3), s=(1, 1))
        self.e2 = CConvBlock(cin=c1, cout=c2, k=(3, 3), s=(1, 2))
        self.e3 = CConvBlock(cin=c2, cout=c3, k=(3, 3), s=(1, 2))

        # Bottleneck: DPRNN + frame-wise full-band attention.
        Cg = 2 * c3   # Treat complex channels as a flat real dimension.
        self.dp = DPRNNBottleneck(
            Cg=Cg,
            hidden_intra=hid_intra,
            hidden_inter=hid_inter,
            num_blocks=num_dp_blocks,
            bidir_intra=bidir_intra,
        )
        self.fba = FramewiseMHSA(
            Cg=Cg, d_model=fba_d_model, nhead=fba_heads, dropout=fba_dropout
        )

        # Decoder: again cin/cout are complex channel counts.
        self.d3 = CDeconvBlock(cin=c3,      cout=c2, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d2 = CDeconvBlock(cin=c2 * 2,  cout=c1, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d1 = CConvBlock   (cin=c1 * 2, cout=c1, k=(3, 3), s=(1, 1))

        # Output mask branches for real and imaginary parts.
        self.out_r = nn.Conv2d(c1, 1, kernel_size=1)
        self.out_i = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _cat_align(a, b):
  
        # 1) Align frequency dimension by trimming to the minimum length.
        F = min(a.size(-1), b.size(-1))
        a = a[..., :F]
        b = b[..., :F]

        # 2) Split into real and imaginary parts.
        ar, ai = torch.chunk(a, 2, dim=1) 
        br, bi = torch.chunk(b, 2, dim=1) 

        # 3) Concatenate real parts and imaginary parts separately.
        real = torch.cat([ar, br], dim=1)  
        imag = torch.cat([ai, bi], dim=1)  

        # 4) Merge back into complex representation.
        return torch.cat([real, imag], dim=1) 

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [B, 2, T, F]
               Complex spectrogram with 1 complex channel (real, imag).

        Returns:
            Complex mask of shape [B, 2, T, F].
        """
        # If you want to support arbitrary c_in, uncomment:
        # x = self.in_proj(x)

        # ----- Encoder -----
        e1 = self.e1(x)   
        e2 = self.e2(e1) 
        e3 = self.e3(e2)    

        # ----- Bottleneck (DPRNN + frame-wise attention) -----
        yb = self.dp(e3)   
        yb = self.fba(yb)  

        # ----- Decoder with complex skip connections -----
        d3 = self.d3(yb)             
        d3 = self._cat_align(d3, e2) 

        d2 = self.d2(d3)             
        d2 = self._cat_align(d2, e1)  

        d1 = self.d1(d2)            

        # ----- Output complex mask -----
        dr, di = torch.chunk(d1, 2, dim=1)  
        mr = torch.tanh(self.out_r(dr))     
        mi = torch.tanh(self.out_i(di))      
        return torch.cat([mr, mi], dim=1)  
