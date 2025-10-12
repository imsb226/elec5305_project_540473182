import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3,3), s=(1,1), p=None):
        super().__init__()
        if p is None: 
            p = (k[0]//2, k[1]//2)
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(cout)
        self.act  = nn.PReLU(cout)
    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3,3), s=(1,1), o=(0,0), p=None):
        super().__init__()
        if p is None: 
            p = (k[0]//2, k[1]//2)
        self.deconv = nn.ConvTranspose2d(cin, cout, k, stride=s, padding=p, output_padding=o, bias=False)
        self.bn     = nn.BatchNorm2d(cout)
        self.act    = nn.PReLU(cout)
    def forward(self, x): 
        return self.act(self.bn(self.deconv(x)))

class GDPRNNBlock(nn.Module):
    """
    Dual-path block: Intra-frame BiGRU then Inter-frame UniGRU with residuals and projections.
    Shapes:
      - Intra (per time step t): sequence length F (frequency axis); input size Cg.
      - Inter (per frequency f): sequence length T (time axis); input size Cg.
    """
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, bidirectional_intra=True):
        super().__init__()
        self.intra = nn.GRU(input_size=Cg, hidden_size=hidden_intra,
                            num_layers=1, batch_first=True,
                            bidirectional=bidirectional_intra)
        intra_out = hidden_intra * (2 if bidirectional_intra else 1)
        self.intra_proj = nn.Linear(intra_out, Cg)
        self.inter = nn.GRU(input_size=Cg, hidden_size=hidden_inter,
                            num_layers=1, batch_first=True,
                            bidirectional=False)
        self.inter_proj = nn.Linear(hidden_inter, Cg)
        self.ln_intra = nn.LayerNorm(Cg)
        self.ln_inter = nn.LayerNorm(Cg)

    def forward(self, xg):
        """
        xg: (B, Cg, T, F)
        """
        B, Cg, T, F = xg.shape

        # Intra-frame along F
        h = xg.permute(0, 2, 3, 1).contiguous().view(B*T, F, Cg)
        h, _ = self.intra(h)
        h = self.intra_proj(h)
        h = self.ln_intra(h)
        h = h.view(B, T, F, Cg).permute(0, 3, 1, 2).contiguous()
        xg = xg + h

        # Inter-frame along T
        z = xg.permute(0, 3, 2, 1).contiguous().view(B*F, T, Cg)
        z, _ = self.inter(z)
        z = self.inter_proj(z)
        z = self.ln_inter(z)
        z = z.view(B, F, T, Cg).permute(0, 3, 2, 1).contiguous()
        xg = xg + z

        return xg

class GDPRNNBottleneck(nn.Module):
    def __init__(self, C, hidden_intra=64, hidden_inter=64, repeat=1, bidirectional_intra=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            GDPRNNBlock(Cg=C, hidden_intra=hidden_intra, hidden_inter=hidden_inter,
                        bidirectional_intra=bidirectional_intra)
            for _ in range(repeat)
        ])

    def forward(self, x):  # x: (B, C, T, F)
        for blk in self.blocks:
            x = blk(x)
        return x

class UNetBiGRU(nn.Module):
    """
    Input:  (B, C_in, T, F)
    Output: (B, 2,    T, F)  # CRM (Real, Imag)
    """
    def __init__(self, c_in=3, base=16, groups=2, hid_intra=48, hid_inter=48, repeat_gdprnn=1):
        super().__init__()
        c1, c2, c3 = base, base*2, base*2

        # Encoder: downsample only along F (stride=1 along T)
        self.e1 = ConvBlock(c_in, c1, k=(1,5), s=(1,2))
        self.e2 = ConvBlock(c1,  c2, k=(3,3), s=(1,2))
        self.e3 = ConvBlock(c2,  c3, k=(3,3), s=(1,2))

        # Bottleneck: G-DPRNN
        self.gdprnn = GDPRNNBottleneck(C=c3,
                                       hidden_intra=hid_intra,
                                       hidden_inter=hid_inter,
                                       repeat=repeat_gdprnn)

        # Decoder with skip connections
        self.d3 = DeconvBlock(c3,   c2, k=(3,3), s=(1,2), o=(0,1))
        self.d2 = DeconvBlock(c2*2, c1, k=(3,3), s=(1,2), o=(0,1))
        self.d1 = DeconvBlock(c1*2, c1, k=(1,5), s=(1,2), o=(0,1), p=(0,2))

        self.out = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x):
        B, C, T, F = x.shape
        x1 = self.e1(x)                 # (B, c1, T, F/2)
        x2 = self.e2(x1)                # (B, c2, T, F/4)
        x3 = self.e3(x2)                # (B, c3, T, F/8)

        x3 = self.gdprnn(x3)            # (B, c3, T, F/8)

        y = self.d3(x3)                 # (B, c2, T, F/4)
        minF = min(y.shape[-1], x2.shape[-1])
        y = torch.cat([y[..., :, :minF], x2[..., :, :minF]], dim=1)

        y = self.d2(y)                  # (B, c1, T, F/2)
        minF = min(y.shape[-1], x1.shape[-1])
        y = torch.cat([y[..., :, :minF], x1[..., :, :minF]], dim=1)

        y = self.d1(y)                  # (B, c1, T, F)
        if y.shape[-1] != F:            # align tail
            y = y[..., :, :F]

        crm = torch.tanh(self.out(y))   # (B, 2, T, F)
        return crm
