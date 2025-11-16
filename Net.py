import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================== Basic Complex-Valued Building Blocks =====================

class ComplexConv2d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None, bias=False):
        """
        Complex 2D convolution.

        Convention:
            - Input channels  = 2 * cin  (real part + imaginary part)
            - Output channels = 2 * cout
        """
        super().__init__()
        if p is None:
            p = (k[0] // 2, k[1] // 2)

        # Real and imaginary convolutions share the same input features,
        # but are applied separately to real and imaginary parts.
        self.real = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=bias)
        self.imag = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=bias)

    def forward(self, x):
        # Split into real and imaginary parts along channel dimension.
        xr, xi = torch.chunk(x, 2, dim=1)
        # (a + j b) * (w_r + j w_i) = (a*w_r - b*w_i) + j(a*w_i + b*w_r)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        # Concatenate real and imaginary parts back together.
        return torch.cat([yr, yi], dim=1)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3), s=(1, 1), p=None, o=(0, 0), bias=False):
        """
        Complex 2D transposed convolution (for upsampling).

        Convention:
            - Input channels  = 2 * cin
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
        BatchNorm for complex feature maps.

        Assumes input channels = 2 * c and applies BN to real/imag parts separately.
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
        PReLU for complex feature maps.

        Assumes input channels = 2 * c and applies PReLU to real/imag parts separately.
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
        Complex convolution block: Conv -> BN -> PReLU.

        cin, cout are complex channel counts (real/imag each have cin/cout channels).
        Actual input channels = 2 * cin, output channels = 2 * cout.
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
        Complex transposed convolution block: Deconv -> BN -> PReLU.

        cin, cout are complex channel counts.
        """
        super().__init__()
        self.deconv = ComplexConvTranspose2d(cin, cout, k, s, p, o, bias=False)
        self.bn = ComplexBN(cout)
        self.act = ComplexPReLU(cout)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


# ===================== DPRNN (Dual-Path RNN) =====================

class DPRNNBlock(nn.Module):
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, bidir_intra=True):
        """
        One DPRNN block with:
            - Intra-GRU: operates along frequency axis for each time frame.
            - Inter-GRU: operates along time axis for each frequency bin.

        Cg is treated as a real-valued feature dimension here.
        """
        super().__init__()
        # Intra-GRU over frequency dimension.
        self.intra = nn.GRU(
            Cg, hidden_intra, num_layers=1, batch_first=True, bidirectional=bidir_intra
        )
        intra_out = hidden_intra * (2 if bidir_intra else 1)
        self.intra_proj = nn.Linear(intra_out, Cg)
        self.ln_intra = nn.LayerNorm(Cg)

        # Inter-GRU over time dimension.
        self.inter = nn.GRU(
            Cg, hidden_inter, num_layers=1, batch_first=True, bidirectional=False
        )
        self.inter_proj = nn.Linear(hidden_inter, Cg)
        self.ln_inter = nn.LayerNorm(Cg)

    def forward(self, xg):
        """
        xg: (B, Cg, T, F), where:
            B = batch size
            Cg = feature channels
            T = time frames
            F = frequency bins
        """
        B, Cg, T, Freq = xg.shape

        # ----- Intra-GRU: model local dependencies across frequency for each time -----
        # Reshape to merge batch and time dimensions for GRU:
        h = (
            xg.permute(0, 2, 3, 1)   # (B, T, F, Cg)
              .contiguous()
              .view(B * T, Freq, Cg)
        )
        h, _ = self.intra(h)
        h = self.intra_proj(h)
        h = self.ln_intra(h)

        # Restore shape back to (B, Cg, T, F).
        h = (
            h.view(B, T, Freq, Cg)
             .permute(0, 3, 1, 2)
             .contiguous()
        )
        xg = xg + h  # Residual connection on frequency modeling.

        # ----- Inter-GRU: model dependencies across time for each frequency bin -----
        z = (
            xg.permute(0, 3, 2, 1)   # (B, F, T, Cg)
              .contiguous()
              .view(B * Freq, T, Cg)
        )
        z, _ = self.inter(z)
        z = self.inter_proj(z)
        z = self.ln_inter(z)

        # Restore shape back to (B, Cg, T, F).
        z = (
            z.view(B, Freq, T, Cg)
             .permute(0, 3, 2, 1)
             .contiguous()
        )
        xg = xg + z  # Residual connection on temporal modeling.

        return xg


class DPRNNBottleneck(nn.Module):
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, num_blocks=2, bidir_intra=True):
        """
        Stack multiple DPRNN blocks as a bottleneck module.
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


# ===================== cfLayerNorm (Channel-Frequency LN) =====================

class CfLayerNorm(nn.Module):
    """
    LayerNorm over (channel, frequency) dimensions.

    Input:
        x: (B, C, T, F)

    For each (B, t), we normalize across (C, F).
    Learnable gamma/beta act on channels and are shared across frequencies,
    following TF-GridNet style cfLN.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # Mean and variance across channel and frequency, for each time step.
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 3), keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta


# ===================== Frame-wise Multi-Head Self-Attention =====================

class FramewiseMHSA(nn.Module):
    """
    Frame-wise multi-head self-attention over the time dimension,
    inspired by TF-GridNet cross-frame self-attention.

    Input:
        x: (B, Cg, T, F)

    Here:
        Cg = D = bottleneck channel count (real-valued features after complex split).
        Self-attention is computed along the time axis for each frequency set.
    """
    def __init__(self, Cg=96, num_heads=4, E=24):
        """
        Args:
            Cg        : input/output channel size (D).
            num_heads : L, number of attention heads.
            E         : query/key embedding size per head (TF-GridNet's E).
                        Value embedding per head uses Dh = Cg / num_heads.
        """
        super().__init__()
        assert Cg % num_heads == 0, "Cg must be divisible by num_heads"

        self.Cg = Cg
        self.L = num_heads
        self.E = E
        self.Dh = Cg // num_heads  # Value dimension per head.

        # Q/K/V are produced by 1x1 convs followed by PReLU + cfLN.
        self.q_conv = nn.Conv2d(Cg, num_heads * E, kernel_size=1)
        self.k_conv = nn.Conv2d(Cg, num_heads * E, kernel_size=1)
        self.v_conv = nn.Conv2d(Cg, num_heads * self.Dh, kernel_size=1)

        self.q_act = nn.PReLU()
        self.k_act = nn.PReLU()
        self.v_act = nn.PReLU()

        self.q_ln = CfLayerNorm(num_heads * E)
        self.k_ln = CfLayerNorm(num_heads * E)
        self.v_ln = CfLayerNorm(num_heads * self.Dh)

        # Output projection after concatenating all heads.
        self.out_conv = nn.Conv2d(Cg, Cg, kernel_size=1)
        self.out_act = nn.PReLU()
        self.out_ln = CfLayerNorm(Cg)

    def forward(self, x):
        """
        x: (B, Cg, T, F)
        """
        B, Cg, T, F = x.shape
        L, E, Dh = self.L, self.E, self.Dh

        # ----- 1) Q, K, V projection -----
        q = self.q_ln(self.q_act(self.q_conv(x)))  # (B, L*E,  T, F)
        k = self.k_ln(self.k_act(self.k_conv(x)))  # (B, L*E,  T, F)
        v = self.v_ln(self.v_act(self.v_conv(x)))  # (B, L*Dh, T, F)

        # ----- 2) Reshape to separate heads -----
        q = q.view(B, L, E, T, F)
        k = k.view(B, L, E, T, F)
        v = v.view(B, L, Dh, T, F)

        # Move time/frequency to middle for convenience: (B, L, T, F, E/Dh).
        q = q.permute(0, 1, 3, 4, 2).contiguous()
        k = k.permute(0, 1, 3, 4, 2).contiguous()
        v = v.permute(0, 1, 3, 4, 2).contiguous()

        # Flatten (F, E) or (F, Dh) so we attend over time only.
        q = q.view(B, L, T, F * E)
        k = k.view(B, L, T, F * E)
        v = v.view(B, L, T, F * Dh)

        # ----- 3) Scaled dot-product attention along time axis -----
        # Scale factor from TF-GridNet: sqrt(L * F * E).
        scale = math.sqrt(L * F * E)

        # Attention logits: for each head, relate all time frames to each other.
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / scale  # (B, L, T, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted sum of V over time.
        attn_out = torch.matmul(attn_weights, v)  # (B, L, T, F*Dh)

        # ----- 4) Restore to (B, Cg, T, F) by merging heads -----
        attn_out = attn_out.view(B, L, T, F, Dh)
        attn_out = attn_out.permute(0, 1, 4, 2, 3).contiguous()  # (B, L, Dh, T, F)
        attn_out = attn_out.view(B, Cg, T, F)

        # ----- 5) Output projection + residual connection -----
        y = self.out_ln(self.out_act(self.out_conv(attn_out)))
        return x + y


# ===================== Main Network: DCCRN + DPRNN + Framewise MHSA =====================

class DCCRN_DPRNN(nn.Module):
    def __init__(
        self,
        base=32,
        hid_intra=96,
        hid_inter=96,
        num_dp_blocks=2,
        bidir_intra=True,

    ):
        """
        DCCRN-style complex encoder-decoder with a DPRNN bottleneck and
        frame-wise attention in the bottleneck.

        The network estimates a complex mask in the STFT domain.
        """
        super().__init__()

        # Encoder complex channel configuration.
        c1, c2, c3 = base, int(base * 1.5), int(base * 1.5)

        # Encoder: cin is complex channel count (1 complex channel = real+imag).
        self.e1 = CConvBlock(cin=1,  cout=c1, k=(3, 3), s=(1, 1))
        self.e2 = CConvBlock(cin=c1, cout=c2, k=(3, 3), s=(1, 2))
        self.e3 = CConvBlock(cin=c2, cout=c3, k=(3, 3), s=(1, 2))

        # Bottleneck: DPRNN + frame-wise multi-head self-attention.
        # Here Cg is the real-valued channel dimension after complex split (2 * c3).
        Cg = 2 * c3
        self.dp = DPRNNBottleneck(
            Cg=Cg,
            hidden_intra=hid_intra,
            hidden_inter=hid_inter,
            num_blocks=num_dp_blocks,
            bidir_intra=bidir_intra,
        )

        self.fba = FramewiseMHSA()

        # Decoder: complex transposed convolutions with skip connections.
        # Note: cin is the complex channel count before split.
        self.d3 = CDeconvBlock(cin=c3 * 2, cout=c2, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d2 = CDeconvBlock(cin=c2 * 2, cout=c1, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d1 = CConvBlock   (cin=c1 * 2, cout=c1, k=(3, 3), s=(1, 1))

        # Final 1x1 convolutions to predict real and imaginary masks separately.
        self.out_r = nn.Conv2d(c1, 1, kernel_size=1)
        self.out_i = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _cat_align(a, b):
        """
        Complex skip-connection concatenation with frequency alignment.

        Inputs:
            a, b: complex tensors of shape (B, 2*Ca, T, Fa) and (B, 2*Cb, T, Fb)

        Steps:
            1. Align along frequency axis by cropping to the minimum frequency.
            2. Split real/imag parts for each tensor.
            3. Concatenate all real parts, then all imaginary parts.
            4. Merge back to a single complex tensor with channel layout:
               [all real channels, all imaginary channels].
        """
        # Align frequency dimension by cropping to the smallest F.
        F = min(a.size(-1), b.size(-1))
        a = a[..., :F]
        b = b[..., :F]

        # Split into real/imag.
        ar, ai = torch.chunk(a, 2, dim=1)
        br, bi = torch.chunk(b, 2, dim=1)

        # Concatenate real and imaginary parts separately.
        real = torch.cat([ar, br], dim=1)
        imag = torch.cat([ai, bi], dim=1)

        # Rebuild complex tensor.
        return torch.cat([real, imag], dim=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 2, T, F), input complex STFT (real + imag in channel dimension).

        Returns:
            Complex mask M_hat: (B, 2, T, F), to be applied to noisy STFT.
        """
        # ----- Encoder -----
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        # ----- Bottleneck: DPRNN + frame-wise attention -----
        yb = self.dp(e3)
        yb = self.fba(yb)

        # Concatenate bottleneck output with encoder output at the same scale.
        yb = self._cat_align(yb, e3)

        # ----- Decoder with complex skip connections -----
        d3 = self.d3(yb)
        d3 = self._cat_align(d3, e2)

        d2 = self.d2(d3)
        d2 = self._cat_align(d2, e1)

        d1 = self.d1(d2)

        # ----- Output complex mask -----
        dr, di = torch.chunk(d1, 2, dim=1)     # Split final complex features.
        mr = torch.tanh(self.out_r(dr))        # Real part of mask.
        mi = torch.tanh(self.out_i(di))        # Imaginary part of mask.

        return torch.cat([mr, mi], dim=1)
