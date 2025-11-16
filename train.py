# ======================================================
# Train + Test Speech Enhancement (for PyCharm)
# Model: DCCRN_DPRNN (from Complex.py)
# ======================================================

import os, glob, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Net import DCCRN_DPRNN
from torch.amp import GradScaler, autocast  # New AMP API
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import re


DATA_ROOT = r"E:\VOICEBANK_ROOT\VOICEBANK_ROOT"
OUT_DIR = r"E:\speech\out"
CKPT_PATH = r"E:\checkpoint\model.pth"
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
TRAIN_CLIP_SECS = 4.0
USE_AMP = True

# ==== Usually do not need to change ====
SR = 16000
N_FFT = 512
HOP = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Validation speakers & early stopping config ====
VAL_SPK = ["226", "230"]   # Use all utterances of speakers 226 and 230 from train as validation set
EARLY_STOP_PATIENCE = 5   # Stop if val loss does not improve for this many epochs
EARLY_STOP_MIN_DELTA = 1e-5

# ---- AMP (keep your previous setting) ----
USE_AMP_THIS_RUN = (USE_AMP and DEVICE == "cuda")
AMP_DTYPE = torch.bfloat16  # If unsupported, change to torch.float16

# -------------------- Hann window cache (once per device) --------------------
_win_cache = {}

def _get_win(device: torch.device):
    """Cache one Hann window per device to avoid repeated .to() overhead."""
    dev = torch.device(device)
    w = _win_cache.get(dev, None)
    if (w is None) or (w.device != dev) or (w.numel() != N_FFT):
        w = torch.hann_window(N_FFT, device=dev)
        _get_win.last_device = dev  # for debugging only
        _win_cache[dev] = w
    return w

# -------------------- STFT / ISTFT (using cached window) --------------------
def stft(x):
    # x: (B, L) float
    return torch.stft(
        x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=_get_win(x.device), return_complex=True, center=True
    )

def istft(S, length):
    # S: (B, F, T) complex
    return torch.istft(
        S, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=_get_win(S.device), length=length, center=True
    )

# -------------------- Utility functions --------------------
def make_features(S):
    # S: (B, F, T) complex -> (B, 2, T, F)
    S_r, S_i = S.real, S.imag
    return torch.stack([S_r.permute(0, 2, 1), S_i.permute(0, 2, 1)], dim=1)

def ensure_len(x, L):
    if len(x) >= L: return x[:L]
    out = np.zeros(L, dtype=x.dtype); out[:len(x)] = x
    return out

def pad_to_multiple(x, multiple):
    return x if x % multiple == 0 else x + (multiple - x % multiple)

def sisnr(x, s, eps=1e-8):
    s_zm, x_zm = s - s.mean(-1, True), x - x.mean(-1, True)
    proj = (torch.sum(x_zm * s_zm, -1, True) / (torch.sum(s_zm ** 2, -1, True) + eps)) * s_zm
    e_noise = x_zm - proj
    ratio = (torch.sum(proj ** 2, -1) + eps) / (torch.sum(e_noise ** 2, -1) + eps)
    return 10 * torch.log10(ratio + eps)

def sdr(x, s, eps=1e-8):
    num = torch.sum(s ** 2, -1)
    den = torch.sum((s - x) ** 2, -1) + eps
    return 10 * torch.log10(num / den)

# -------------------- Dataset --------------------
class PairList:
    """
    Pair noisy_dir / clean_dir wav files by filename.
    Support include_spk / exclude_spk by speaker ID.
    Speaker ID is parsed from the first digit sequence in the filename, e.g. p226_001.wav -> 226.
    """
    def __init__(self, noisy_dir, clean_dir, include_spk=None, exclude_spk=None):
        self.pairs = []
        include_spk = set(str(s) for s in include_spk) if include_spk is not None else None
        exclude_spk = set(str(s) for s in exclude_spk) if exclude_spk is not None else None

        for n in sorted(glob.glob(os.path.join(noisy_dir, "*.wav"))):
            name = os.path.basename(n)
            c = os.path.join(clean_dir, name)
            if not os.path.exists(c):
                continue

            # Parse speaker ID: first digit sequence in filename
            m = re.search(r"(\d+)", name)
            spk_id = m.group(1) if m else None

            if include_spk is not None:
                # Keep only speakers in include_spk
                if spk_id not in include_spk:
                    continue
            if exclude_spk is not None:
                # Remove speakers in exclude_spk
                if spk_id in exclude_spk:
                    continue

            self.pairs.append((n, c))

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

class VCTKDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, secs=4.0, train=True,
                 include_spk=None, exclude_spk=None):
        """
        include_spk / exclude_spk: list of speaker IDs (strings), e.g. ["226", "230"].
        - Train set: typically use exclude_spk to remove validation speakers.
        - Val / Test: typically use include_spk to keep only desired speakers.
        """
        self.pl = PairList(noisy_dir, clean_dir,
                           include_spk=include_spk,
                           exclude_spk=exclude_spk)
        self.secs, self.train = secs, train
        self.target_len = int(SR * secs)

    def __len__(self): return len(self.pl)

    def __getitem__(self, idx):
        n_path, c_path = self.pl[idx]
        noisy, _ = sf.read(n_path, dtype="float32")
        clean, _ = sf.read(c_path, dtype="float32")
        if noisy.ndim == 2: noisy = noisy[:, 0]
        if clean.ndim == 2: clean = clean[:, 0]
        L = min(len(noisy), len(clean))
        noisy, clean = noisy[:L], clean[:L]

        if self.train:
            # Train: random crop / pad to fixed length
            if L < self.target_len:
                noisy = ensure_len(noisy, self.target_len)
                clean = ensure_len(clean, self.target_len)
            else:
                st = random.randint(0, L - self.target_len)
                noisy, clean = noisy[st:st + self.target_len], clean[st:st + self.target_len]

        # Val / Test: keep original length; pad later in collate_test
        return torch.from_numpy(noisy), torch.from_numpy(clean), os.path.basename(n_path)

def collate_train(b):
    n, c, _ = zip(*b)
    return torch.stack(n), torch.stack(c)

def collate_test(b):
    out = []
    for n, c, name in b:
        L = max(len(n), len(c))
        Lp = pad_to_multiple(L, HOP)
        n = ensure_len(n.numpy(), Lp)
        c = ensure_len(c.numpy(), Lp)
        out.append((torch.from_numpy(n), torch.from_numpy(c), name, L))
    return out

# -------------------- Training --------------------
def train_one_epoch(model, loader, opt, scaler=None):
    model.train()
    tot_loss = 0.0
    tot_si   = 0.0
    seen     = 0

    for noisy, clean in loader:
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)

        use_amp = (scaler is not None) and noisy.is_cuda
        with autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
            # 1) STFT
            S_noisy = stft(noisy)     # complex, (B, F, T)
            S_clean = stft(clean)

            # 2) Build features (B, 2, T, F)
            feats = make_features(S_noisy)

            # 3) Forward
            M_hat = model(feats)           # (B, 2, T, F) in [-1, 1]

            # 4) Complex mask + spectral estimation (in float32 to avoid slow ComplexHalf kernels)
            with autocast('cuda', enabled=False):
                Mr = M_hat[:, 0].permute(0, 2, 1).contiguous().float()
                Mi = M_hat[:, 1].permute(0, 2, 1).contiguous().float()

                S_noisy32 = S_noisy.to(torch.complex64)
                S_clean32 = S_clean.to(torch.complex64)

                M_hat_c = torch.complex(Mr, Mi)                  # complex64, (B, F, T)
                S_hat   = M_hat_c * S_noisy32                    # (B, F, T)

                mag_hat = torch.abs(S_hat).clamp_min(1e-12).pow(0.3)
                mag_ref = torch.abs(S_clean32).clamp_min(1e-12).pow(0.3)
                loss_mag = torch.mean((mag_hat - mag_ref) ** 2)

                est = istft(S_hat, length=noisy.shape[-1])       # (B, L)
                loss_wav = torch.mean(torch.abs(est - clean.float()))

                loss = 0.5 * loss_mag + 0.5 * loss_wav

        opt.zero_grad(set_to_none=True)

        if scaler is not None and noisy.is_cuda:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        bsz = noisy.size(0)
        with torch.no_grad():
            si = sisnr(est, clean).mean().item()
            tot_si   += si   * bsz
            tot_loss += loss.item() * bsz
            seen     += bsz

    avg_loss = tot_loss / max(1, seen)
    avg_si   = tot_si   / max(1, seen)
    return avg_loss, avg_si

# -------------------- Validation (for early stopping) --------------------
@torch.no_grad()
def validate_one_epoch(model, loader):
    """
    Compute the same loss (spectral + waveform) and SiSNR on the validation set.
    loader uses collate_test, batch_size=1.
    """
    model.eval()
    tot_loss = 0.0
    tot_si   = 0.0
    seen     = 0

    for batch in loader:
        # batch_size=1, collate_test returns list[(noisy, clean, name, L)]
        if isinstance(batch, list):
            noisy, clean, name, L_raw = batch[0]
        else:
            noisy, clean, name, L_raw = batch

        if noisy.dim() == 1: noisy = noisy.unsqueeze(0)
        if clean.dim() == 1: clean = clean.unsqueeze(0)

        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)

        # STFT
        S_noisy = stft(noisy)
        S_clean = stft(clean)

        feats = make_features(S_noisy)
        M_hat = model(feats)  # (1, 2, T, F)

        Mr = M_hat[:, 0].permute(0, 2, 1).contiguous().float()
        Mi = M_hat[:, 1].permute(0, 2, 1).contiguous().float()

        S_noisy32 = S_noisy.to(torch.complex64)
        S_clean32 = S_clean.to(torch.complex64)

        M_hat_c = torch.complex(Mr, Mi)
        S_hat   = M_hat_c * S_noisy32

        mag_hat = torch.abs(S_hat).clamp_min(1e-12).pow(0.3)
        mag_ref = torch.abs(S_clean32).clamp_min(1e-12).pow(0.3)
        loss_mag = torch.mean((mag_hat - mag_ref) ** 2)

        est = istft(S_hat, length=noisy.shape[-1])
        loss_wav = torch.mean(torch.abs(est - clean.float()))

        loss = 0.5 * loss_mag + 0.5 * loss_wav

        bsz = noisy.size(0)
        si  = sisnr(est, clean).mean().item()

        tot_loss += loss.item() * bsz
        tot_si   += si * bsz
        seen     += bsz

    avg_loss = tot_loss / max(1, seen)
    avg_si   = tot_si   / max(1, seen)
    return avg_loss, avg_si

# -------------------- Test, save enhanced wav + PESQ/STOI --------------------
@torch.no_grad()
def test_and_save(model, loader, out_dir):
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Metrics
    pesq_metric = PerceptualEvaluationSpeechQuality(16000, mode="wb").to(DEVICE)
    stoi_metric = ShortTimeObjectiveIntelligibility(16000, extended=False).to(DEVICE)

    tot_si = tot_sd = tot_pesq = tot_stoi = 0.0
    count = 0

    for batch in loader:
        if isinstance(batch, list):
            noisy, clean, name, L_raw = batch[0]
        elif isinstance(batch, (tuple, list)) and len(batch) == 4:
            noisy, clean, name, L_raw = batch
        else:
            raise ValueError("Unexpected batch format for batch_size=1")

        if noisy.dim() == 1: noisy = noisy.unsqueeze(0)
        if clean.dim() == 1: clean = clean.unsqueeze(0)

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        S_noisy = stft(noisy)
        feats   = make_features(S_noisy)
        M_hat   = model(feats)

        Mr    = M_hat[:, 0].permute(0, 2, 1).float()
        Mi    = M_hat[:, 1].permute(0, 2, 1).float()
        S_hat = torch.complex(Mr, Mi) * S_noisy.to(torch.complex64)
        est   = istft(S_hat, length=noisy.shape[-1])

        Li   = int(L_raw) if not torch.is_tensor(L_raw) else int(L_raw.item())
        esti = est[0, :Li]
        clni = clean[0, :Li]

        si   = sisnr(esti.unsqueeze(0), clni.unsqueeze(0)).item()
        sd   = sdr(  esti.unsqueeze(0), clni.unsqueeze(0)).item()
        pesq = pesq_metric(esti.unsqueeze(0), clni.unsqueeze(0)).item()
        stoi = stoi_metric(esti.unsqueeze(0), clni.unsqueeze(0)).item()

        tot_si += si; tot_sd += sd
        tot_pesq += pesq; tot_stoi += stoi
        count += 1

        name_i = name if isinstance(name, str) else str(name)
        if not name_i.lower().endswith(".wav"): name_i += ".wav"
        sf.write(os.path.join(out_dir, name_i),
                 esti.detach().cpu().to(torch.float32).numpy(), SR)

    avg_si   = tot_si   / max(1, count)
    avg_sd   = tot_sd   / max(1, count)
    avg_pesq = tot_pesq / max(1, count)
    avg_stoi = tot_stoi / max(1, count)

    print(f"[Metrics] PESQ={avg_pesq:.3f}, STOI={avg_stoi:.3f}, "
          f"SiSNR={avg_si:.2f} dB, SDR={avg_sd:.2f} dB")

    return avg_si, avg_sd

def count_params(model, only_trainable=True):
    ps = [p.numel() for p in model.parameters() if (p.requires_grad or not only_trainable)]
    n = sum(ps)
    def pretty(x):
        if x >= 1_000_000: return f"{x / 1_000_000:.3f} M"
        if x >= 1_000:     return f"{x / 1_000:.3f} K"
        return str(x)
    print(f"Parameters ({'trainable' if only_trainable else 'all'}): {n}  ({pretty(n)})")
    return n

# -------------------- Main --------------------
def main():
    print(f"Device={DEVICE}, AMP={USE_AMP_THIS_RUN}")
    train_noisy = os.path.join(DATA_ROOT, "train", "noisy")
    train_clean = os.path.join(DATA_ROOT, "train", "clean")
    test_noisy  = os.path.join(DATA_ROOT, "test", "noisy")
    test_clean  = os.path.join(DATA_ROOT, "test", "clean")

    # Train set: exclude speakers 226 / 230
    ds_tr = VCTKDataset(
        train_noisy, train_clean,
        secs=TRAIN_CLIP_SECS, train=True,
        include_spk=None,
        exclude_spk=VAL_SPK,
    )

    # Validation set: use only speakers 226 / 230, full utterances (train=False)
    ds_val = VCTKDataset(
        train_noisy, train_clean,
        secs=TRAIN_CLIP_SECS, train=False,
        include_spk=VAL_SPK,
        exclude_spk=None,
    )

    # Test set: original test directory
    ds_te = VCTKDataset(
        test_noisy, test_clean,
        secs=TRAIN_CLIP_SECS, train=False,
        include_spk=None,
        exclude_spk=None,
    )

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=2, collate_fn=collate_train)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_test)
    dl_te = DataLoader(ds_te, batch_size=1, shuffle=False,
                       num_workers=2, collate_fn=collate_test)

    model = DCCRN_DPRNN().to(DEVICE)
    count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    scaler = GradScaler('cuda', enabled=USE_AMP_THIS_RUN)

    # Warm up ISTFT / window once to reduce first-step jitter
    _ = istft(torch.zeros(1, N_FFT // 2 + 1, 4, dtype=torch.complex64, device=DEVICE),
              length=4 * HOP)

    print(f"Train pairs={len(ds_tr)}, Val pairs={len(ds_val)}, Test pairs={len(ds_te)}")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_sis = train_one_epoch(model, dl_tr, opt, scaler)
        val_loss, val_sis = validate_one_epoch(model, dl_val)
        sch.step()

        print(f"[Epoch {ep:02d}] "
              f"train_loss={tr_loss:.4f}, train_SiSNR={tr_sis:.2f} dB | "
              f"val_loss={val_loss:.4f}, val_SiSNR={val_sis:.2f} dB, "
              f"lr={opt.param_groups[0]['lr']:.2e}")

        # ----- Early stopping -----
        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = ep
            epochs_no_improve = 0

            # Save best model whenever validation loss improves
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            torch.save(
                {"model": model.state_dict(),
                 "epoch": ep,
                 "best_val_loss": best_val_loss},
                CKPT_PATH,
            )
            print(f"  -> Val loss improved, checkpoint saved (epoch {ep}, "
                  f"best_val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No val improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {ep} "
                      f"(no val loss improvement in {EARLY_STOP_PATIENCE} epochs).")
                break

    # Load the best model (on validation set) for final testing
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best model from epoch {ckpt.get('epoch', '?')} "
              f"(best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f})")
    else:
        print("WARNING: best checkpoint not found, using last-epoch model.")

    avg_si, avg_sd = test_and_save(model, dl_te, OUT_DIR)
    print(f"Model checkpoint path: {CKPT_PATH}")
    print(f"Enhanced audio saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
