# Inference-only speech enhancement with DCCRN_DPRNN

import os, glob, torch, numpy as np, soundfile as sf
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Net import DCCRN_DPRNN
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from pathlib import Path
# basic config
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "test"
OUT_DIR   =  ROOT / "out"
CKPT_PATH = ROOT / "model.pth"

SR    = 16000
N_FFT = 512
HOP   = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# cached Hann window
_win_cache = {}

def _get_win(device: torch.device):
    # return a cached Hann window on the given device
    dev = torch.device(device)
    w = _win_cache.get(dev, None)
    if (w is None) or (w.device != dev) or (w.numel() != N_FFT):
        w = torch.hann_window(N_FFT, device=dev)
        _win_cache[dev] = w
    return w

# STFT / ISTFT
def stft(x: torch.Tensor) -> torch.Tensor:
    # (B, L) -> (B, F, T) complex STFT
    return torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=_get_win(x.device),
        return_complex=True,
        center=True,
    )

def istft(S: torch.Tensor, length: int) -> torch.Tensor:
    # (B, F, T) -> (B, length) waveform
    return torch.istft(
        S,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=_get_win(S.device),
        length=length,
        center=True,
    )

# small helpers
def ensure_len(x: np.ndarray, L: int) -> np.ndarray:
    # crop or zero-pad 1D array to length L
    if len(x) >= L:
        return x[:L]
    out = np.zeros(L, dtype=x.dtype)
    out[:len(x)] = x
    return out

def pad_to_multiple(x: int, multiple: int) -> int:
    # round x up to the nearest multiple
    return x if x % multiple == 0 else x + (multiple - x % multiple)

def make_features(S: torch.Tensor) -> torch.Tensor:
    # (B, F, T) complex -> (B, 2, T, F) [real, imag]
    S_r, S_i = S.real, S.imag
    return torch.stack(
        [S_r.permute(0, 2, 1), S_i.permute(0, 2, 1)],
        dim=1,
    )

def sisnr(x: torch.Tensor, s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # scale-invariant SNR (dB)
    s_zm = s - s.mean(-1, keepdim=True)
    x_zm = x - x.mean(-1, keepdim=True)
    proj = (
        torch.sum(x_zm * s_zm, -1, keepdim=True)
        / (torch.sum(s_zm ** 2, -1, keepdim=True) + eps)
    ) * s_zm
    e_noise = x_zm - proj
    ratio = (torch.sum(proj ** 2, -1) + eps) / (torch.sum(e_noise ** 2, -1) + eps)
    return 10 * torch.log10(ratio + eps)

def sdr(x: torch.Tensor, s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # plain SDR (dB)
    num = torch.sum(s ** 2, -1)
    den = torch.sum((s - x) ** 2, -1) + eps
    return 10 * torch.log10(num / den)

# dataset
class PairList:
    # pair noisy/clean wav files by filename
    def __init__(self, noisy_dir: str, clean_dir: str):
        self.pairs = []
        for n in sorted(glob.glob(os.path.join(noisy_dir, "*.wav"))):
            name = os.path.basename(n)
            c = os.path.join(clean_dir, name)
            if os.path.exists(c):
                self.pairs.append((n, c))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

class VCTKTest(Dataset):
    # returns (noisy, clean, filename, original_length)
    def __init__(self, noisy_dir: str, clean_dir: str):
        self.pl = PairList(noisy_dir, clean_dir)

    def __len__(self):
        return len(self.pl)

    def __getitem__(self, idx):
        n_path, c_path = self.pl[idx]
        noisy, _ = sf.read(n_path, dtype="float32")
        clean, _ = sf.read(c_path, dtype="float32")

        # if stereo, keep first channel
        if noisy.ndim == 2:
            noisy = noisy[:, 0]
        if clean.ndim == 2:
            clean = clean[:, 0]

        L = max(len(noisy), len(clean))
        Lp = pad_to_multiple(L, HOP)
        noisy = ensure_len(noisy, Lp)
        clean = ensure_len(clean, Lp)

        return torch.from_numpy(noisy), torch.from_numpy(clean), os.path.basename(n_path), L

def collate_test(batch):
    # keep as list (batch_size=1)
    return batch

# checkpoint
def load_checkpoint(model: nn.Module, ckpt_path: str, device=DEVICE, strict: bool = False):
    # load weights into model from ckpt_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)
    state = obj.get("model", obj)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[Load] loaded from {ckpt_path}")
    if missing:
        print(f"[Load] missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[Load] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

# eval + save
@torch.no_grad()
def test_and_save(model: nn.Module, loader, out_dir: str):
    # run inference and save enhanced wavs
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pesq_metric = PerceptualEvaluationSpeechQuality(SR, mode="wb").to(DEVICE)
    stoi_metric = ShortTimeObjectiveIntelligibility(SR, extended=False).to(DEVICE)

    tot_si = tot_sd = tot_pesq = tot_stoi = 0.0
    count = 0

    for batch in loader:
        # batch is [ (noisy, clean, name, L_raw) ]
        if isinstance(batch, list):
            noisy, clean, name, L_raw = batch[0]
        else:
            noisy, clean, name, L_raw = batch

        if noisy.dim() == 1:
            noisy = noisy.unsqueeze(0)
        if clean.dim() == 1:
            clean = clean.unsqueeze(0)

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        # STFT -> model
        S_noisy = stft(noisy)             # (1, F, T)
        feats   = make_features(S_noisy)  # (1, 2, T, F)
        M_hat   = model(feats)            # (1, 2, T, F)

        # complex mask
        Mr = M_hat[:, 0].permute(0, 2, 1).float()  # (1, F, T)
        Mi = M_hat[:, 1].permute(0, 2, 1).float()
        S_hat = torch.complex(Mr, Mi) * S_noisy.to(torch.complex64)
        est   = istft(S_hat, length=noisy.shape[-1])  # (1, L)


        # back to original length
        Li = int(L_raw) if not torch.is_tensor(L_raw) else int(L_raw.item())
        esti = est[0, :Li]
        clni = clean[0, :Li]

        # metrics
        si   = sisnr(esti.unsqueeze(0), clni.unsqueeze(0)).item()
        sd   = sdr(  esti.unsqueeze(0), clni.unsqueeze(0)).item()
        pesq = pesq_metric(esti.unsqueeze(0), clni.unsqueeze(0)).item()
        stoi = stoi_metric(esti.unsqueeze(0), clni.unsqueeze(0)).item()

        tot_si   += si
        tot_sd   += sd
        tot_pesq += pesq
        tot_stoi += stoi
        count    += 1

        # save wav
        name_i = name if isinstance(name, str) else str(name)
        if not name_i.lower().endswith(".wav"):
            name_i += ".wav"
        sf.write(
            os.path.join(out_dir, name_i),
            esti.detach().cpu().to(torch.float32).numpy(),
            SR,
        )

    avg_si   = tot_si   / max(1, count)
    avg_sd   = tot_sd   / max(1, count)
    avg_pesq = tot_pesq / max(1, count)
    avg_stoi = tot_stoi / max(1, count)

    print(
        f"[Metrics] PESQ={avg_pesq:.3f}, STOI={avg_stoi:.3f}, "
        f"SiSNR={avg_si:.2f} dB, SDR={avg_sd:.2f} dB"
    )
    print(f"[Done] enhanced wavs saved to: {out_dir}")

# main
def main():
    print(f"Device: {DEVICE}")
    test_noisy = os.path.join(DATA_ROOT, "test", "noisy")
    test_clean = os.path.join(DATA_ROOT, "test", "clean")

    ds_te = VCTKTest(test_noisy, test_clean)
    dl_te = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_test,
    )

    model = DCCRN_DPRNN().to(DEVICE)

    # warm up cuFFT once
    _ = istft(
        torch.zeros(
            1,
            N_FFT // 2 + 1,
            4,
            dtype=torch.complex64,
            device=DEVICE,
        ),
        length=4 * HOP,
    )

    load_checkpoint(model, CKPT_PATH, device=DEVICE, strict=False)
    test_and_save(model, dl_te, OUT_DIR)

if __name__ == "__main__":
    main()
