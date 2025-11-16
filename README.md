## Speech Enhancement (Lightweight, CPU-Friendly)

Single-channel speech enhancement model targeting **laptop-CPU deployment** with **< 1M parameters**, while maintaining practical perceptual and intelligibility quality.  
Input is complex STFT; the model predicts a **Complex Ratio Mask (CRM)** and reconstructs waveforms via iSTFT.

Backbone: a U-Net–style encoder/decoder that downsamples **only along the frequency axis**, with a **dual-path recurrent bottleneck** (intra-frequency BiGRU + inter-time GRU, both with projection, residual connections, and LayerNorm).

---

## Status

**Completed.**  
Code and experiments correspond to the project report and the results listed below.

---

## Goals

- PESQ (wb) ≥ 2.95  
- STOI ≥ 0.94  
- ΔSI-SDR ≥ 6.0 dB  
- Parameters < 1.0 M and **CPU-friendly latency**

---

## Dataset and Protocol

- **Dataset**: VoiceBank+DEMAND (16 kHz, standard split and SNR settings)  
- **Task**: single-channel noisy → clean speech mapping using paired training  



---

## Model Overview

### STFT and Masking

- 16 kHz audio  
- STFT with `n_fft = 512`, `hop = 256`, Hann window  
- Model input: noisy complex STFT \(Y(f, k)\), split into real/imag channels  
- Model output: 2-channel CRM \([\Re(M), \Im(M)]\)  
- Enhanced spectrum: \(\hat{S}(f, k) = M(f,k)\,Y(f,k)\), followed by iSTFT

### Encoder–Decoder (Complex U-Net)

- **Frequency-only downsampling** to preserve temporal resolution  
- Three complex encoder blocks:
  - ComplexConv2d → complex BatchNorm → complex PReLU  
  - Stride 2 along frequency, stride 1 along time  
- Three symmetric complex decoder blocks:
  - ComplexConvTranspose2d → complex BatchNorm → complex PReLU  
  - Skip connections from encoder to decoder at each scale  
- Complex convolutions follow the usual decomposition and jointly model real and imaginary parts, instead of treating them as independent real channels.

### Dual-Path Recurrent Bottleneck (DPCRN-style)

- Bottleneck feature shape: \([B, C_g, T, F]\), where \(C_g\) stacks real and imaginary channels  
- **Intra-frequency BiGRU**:
  - For each time frame, run a BiGRU over the frequency axis  
  - Followed by linear projection, LayerNorm, residual connection  
- **Inter-time GRU**:
  - For each frequency bin, run a GRU over the time axis  
  - Followed by projection, LayerNorm, residual connection  

This dual-path structure provides long-range **spectral and temporal context** with relatively small hidden sizes.

### Frame-wise Self-Attention (Bottleneck)

- One **frame-wise multi-head self-attention block** at the bottleneck:
  - Treat each time frame as a token (frequency + channels are folded into the feature dimension)  
  - Attention is computed over the time axis only → complexity \(O(T^2)\) instead of \(O((TF)^2)\)  
- Implemented as:
  - 1×1 convs for Q/K/V → reshape to \([T, \cdot]\) per head  
  - Multi-head attention over frames  
  - 1×1 conv + LayerNorm + residual  

Empirically this block provides a clear gain in PESQ and ΔSI-SDR with only a **small** increase in parameters.

---

## Training Objective

- **Spectral loss** on compressed magnitudes:  
  MSE between \(|\hat{S}|^\gamma\) and \(|S|^\gamma\), with \(\gamma = 0.3\)  
- **Waveform loss**:  
  L1 loss between enhanced and clean waveforms  
- Total loss:  
  \(\mathcal{L} = \alpha \mathcal{L}_\text{mag} + (1-\alpha)\mathcal{L}_\text{wav}\), with \(\alpha = 0.5\)

---

## Results (VoiceBank+DEMAND)

Final model:

- **PESQ (wb)**: 3.035  
- **STOI**: 94.8 %  
- **ΔSI-SDR**: 10.68 dB  
- **Parameters**: 0.676 M  

### Ablation Highlights

- **No complex convolution** (treat real/imag as separate real channels):
  - Params ↓ to 0.411 M  
  - Small but consistent drop in PESQ / ΔSI-SDR  
  - → Complex conv helps, but the gain is modest relative to its parameter cost  
- **No frame-wise self-attention** (keep complex convs, remove attention):
  - Params ≈ 0.638 M  
  - More noticeable drop in PESQ / ΔSI-SDR  
  - → Bottleneck attention is a **more cost-effective** component in this setting

---

## Future Work

- Explore lighter or partially shared complex convolutions  
- Reallocate saved parameters to bottleneck / attention modules  
- Add explicit CPU latency / RTF benchmarks  
- Evaluate on additional noisy corpora
