import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPE1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:(d_model//2)])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x: [B, L, C]
        L = x.size(1)
        return x + self.pe[:L, :].unsqueeze(0).to(x.device, x.dtype)

class FreqTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, ff_mult: int = 2, dropout: float = 0.0, layers: int = 1):
        super().__init__()
        self.pe = SinusoidalPE1D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_mult*d_model,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x):  # [B, F, Cg]
        x = self.pe(x)
        y = self.encoder(x)
        return self.out_ln(y)

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3,3), s=(1,1), p=None):
        super().__init__()
        if p is None: p = (k[0]//2, k[1]//2)
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(cout)
        self.act  = nn.PReLU(cout)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3,3), s=(1,1), o=(0,0), p=None):
        super().__init__()
        if p is None: p = (k[0]//2, k[1]//2)
        self.deconv = nn.ConvTranspose2d(cin, cout, k, stride=s, padding=p, output_padding=o, bias=False)
        self.bn     = nn.BatchNorm2d(cout)
        self.act    = nn.PReLU(cout)
    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

class GDPRNNBlock(nn.Module):
    def __init__(self, Cg, hidden_intra=64, hidden_inter=64, bidirectional_intra=True,
                 use_freq_transformer: bool = True, freq_tf_nhead: int = 4,
                 freq_tf_ff_mult: int = 2, freq_tf_dropout: float = 0.0, freq_tf_layers: int = 1):
        super().__init__()
        self.use_freq_transformer = use_freq_transformer
        if use_freq_transformer:
            self.freq_tf = FreqTransformer(
                d_model=Cg, nhead=freq_tf_nhead, ff_mult=freq_tf_ff_mult,
                dropout=freq_tf_dropout, layers=freq_tf_layers
            )

        self.intra = nn.GRU(input_size=Cg, hidden_size=hidden_intra, num_layers=1,
                            batch_first=True, bidirectional=bidirectional_intra)
        intra_out = hidden_intra * (2 if bidirectional_intra else 1)
        self.intra_proj = nn.Linear(intra_out, Cg)
        self.ln_intra = nn.LayerNorm(Cg)

        self.inter = nn.GRU(input_size=Cg, hidden_size=hidden_inter, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.inter_proj = nn.Linear(hidden_inter, Cg)
        self.ln_inter = nn.LayerNorm(Cg)

    def forward(self, xg):
        B, Cg, T, Freq = xg.shape

        h = xg.permute(0, 2, 3, 1).contiguous().view(B*T, Freq, Cg)
        if self.use_freq_transformer:
            h = self.freq_tf(h)

        h, _ = self.intra(h)
        h = self.intra_proj(h)
        h = self.ln_intra(h)
        h = h.view(B, T, Freq, Cg).permute(0, 3, 1, 2).contiguous()
        xg = xg + h

        z = xg.permute(0, 3, 2, 1).contiguous().view(B*Freq, T, Cg)
        z, _ = self.inter(z)
        z = self.inter_proj(z)
        z = self.ln_inter(z)
        z = z.view(B, Freq, T, Cg).permute(0, 3, 2, 1).contiguous()
        xg = xg + z
        return xg

class GDPRNNBottleneck(nn.Module):
    def __init__(self, C, hidden_intra=64, hidden_inter=64, repeat=1, bidirectional_intra=True):
        super().__init__()
        self.block = GDPRNNBlock(Cg=C, hidden_intra=hidden_intra, hidden_inter=hidden_inter,
                                 bidirectional_intra=bidirectional_intra)
    def forward(self, x):
        x = self.block(x)
        return x

class UNetBiGRU(nn.Module):
    def __init__(self, c_in=3, base=16, groups=2, hid_intra=48, hid_inter=48, repeat_gdprnn=1):
        super().__init__()
        c1, c2, c3 = base, base*2, base*2
        self.e1 = ConvBlock(c_in, c1, k=(3,3), s=(1,1))
        self.e2 = ConvBlock(c1,  c2, k=(3,3), s=(1,2))
        self.e3 = ConvBlock(c2,  c3, k=(3,3), s=(1,2))

        self.gdprnn1 = GDPRNNBottleneck(C=c3, hidden_intra=hid_intra, hidden_inter=hid_inter, repeat=repeat_gdprnn)
        self.gdprnn2 = GDPRNNBottleneck(C=c3, hidden_intra=hid_intra, hidden_inter=hid_inter, repeat=repeat_gdprnn)
        self.gdprnn3 = GDPRNNBottleneck(C=c3, hidden_intra=hid_intra, hidden_inter=hid_inter, repeat=repeat_gdprnn)

        self.d3 = DeconvBlock(c3,   c2, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d2 = DeconvBlock(c2*2, c1, k=(3, 3), s=(1, 2), o=(0, 1))
        self.d1 = DeconvBlock(c1*2, c1, k=(3, 3), s=(1, 1), o=(0, 0))
        self.out = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x):
        B, C, T, Freq = x.shape
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)

        x3 = self.gdprnn1(x3)
        x3 = self.gdprnn2(x3)
        x3 = self.gdprnn3(x3)

        y = self.d3(x3)
        minF = min(y.shape[-1], x2.shape[-1])
        y = torch.cat([y[..., :, :minF], x2[..., :, :minF]], dim=1)
        y = self.d2(y)
        minF = min(y.shape[-1], x1.shape[-1])
        y = torch.cat([y[..., :, :minF], x1[..., :, :minF]], dim=1)
        y = self.d1(y)
        if y.shape[-1] != Freq:
            y = y[..., :, :Freq]
        crm = torch.tanh(self.out(y))
        return crm

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C_in, T, Freq = 2, 3, 64, 256
    x = torch.randn(B, C_in, T, Freq)
    net = UNetBiGRU(c_in=C_in, base=16, hid_intra=48, hid_inter=48, repeat_gdprnn=1)
    with torch.no_grad():
        y = net(x)
    print("input :", x.shape)
    print("output:", y.shape)
