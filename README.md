# Lightweight STFT-Based Speech Enhancement

Single-channel speech enhancement model targeting **laptop-CPU deployment** with **< 1M parameters**, while maintaining practical perceptual and intelligibility quality.

The model operates in the short-time Fourier transform (STFT) domain: it takes the noisy complex STFT as input, predicts a **complex ratio mask (CRM)**, applies it to the mixture, and reconstructs the enhanced waveform using inverse STFT (iSTFT).

---

## Status

**Completed.**  
Code and experiments correspond to the project report and the results listed below.

---

## Goals

- PESQ (wideband) ≥ 2.95  
- STOI ≥ 0.94  
- Delta SI-SDR ≥ 6.0 dB  
- Total parameters < 1.0 M 

---

## Dataset and Protocol

- **Dataset**: VoiceBank+DEMAND, 16 kHz, standard train/test split and SNR settings  
- **Task**: single-channel noisy → clean mapping using paired training  
- **Metrics**:
  - PESQ (wideband)  
  - STOI  
  - Delta SI-SDR (improvement over the noisy mixture)


---

## Model Overview

### STFT and Masking

- Sample rate: 16 kHz  
- STFT: `n_fft = 512`, `hop = 256`, Hann window  
- Input: noisy complex STFT `Y(f, k)`, split into two channels (real part and imaginary part)  
- Output: a 2-channel complex ratio mask (CRM), `[Re(M), Im(M)]`  
- Enhanced spectrum:  
  `S_hat(f, k) = M(f, k) * Y(f, k)`  
- Waveform reconstruction: iSTFT applied to `S_hat(f, k)` to obtain the enhanced waveform

### Encoder–Decoder (Complex U-Net)

The encoder–decoder follows a U-Net style architecture that **only downsamples along the frequency axis** and keeps the original temporal resolution.

- **Encoder**:
  - Three complex convolution blocks
  - Each block: complex Conv2d on the stacked real/imag channels, complex BatchNorm (separate normalisation for real and imaginary parts), and complex PReLU
  - Frequency is downsampled with stride 2, time stride is 1

- **Decoder**:
  - Three complex transposed convolution blocks mirroring the encoder
  - Complex ConvTranspose2d + complex BatchNorm + complex PReLU
  - Skip connections from encoder to decoder at the same scale to preserve fine spectral detail

Real and imaginary parts of the STFT are always processed jointly through complex convolutions, instead of being treated as two completely independent real-valued channels.

### Dual-Path Recurrent Bottleneck (DPCRN-style)

Between the encoder and decoder, the model uses a **dual-path recurrent bottleneck** inspired by DPCRN/DPRNN. The bottleneck feature has shape `[B, Cg, T, F]`, where:

- `B` = batch size  
- `Cg` = channel dimension after stacking real and imaginary parts  
- `T` = number of time frames  
- `F` = downsampled frequency bins  

The bottleneck consists of two recurrent stages:

- **Intra-frequency BiGRU**:
  - For each time frame, a bidirectional GRU runs along the frequency axis
  - Followed by a linear projection, LayerNorm and a residual connection back to the input

- **Inter-time GRU**:
  - For each frequency bin, a GRU runs along the time axis
  - Again followed by projection, LayerNorm and residual connection

This dual-path design allows the model to capture long-range dependencies in both frequency (e.g., harmonics, formants) and time (e.g., sustained vowels, long noise segments) with relatively small hidden sizes, which is important for keeping the model lightweight.

### Frame-wise Self-Attention (Bottleneck)

On top of the dual-path bottleneck, a single **frame-wise multi-head self-attention** block is applied at the bottleneck:

- Each time frame is treated as a token; frequency bins and channels are folded into a feature vector per frame  
- Attention is computed **only over the time axis**, so the complexity scales roughly as `O(T^2)` instead of `O((T * F)^2)`  
- Implementation outline:
  - 1×1 convolutions generate Q, K and V from the bottleneck feature
  - Q, K and V are reshaped into `[num_heads, T, feature_per_head]`
  - Standard multi-head self-attention is applied over frames
  - The result is projected back with a 1×1 convolution, followed by LayerNorm and a residual connection

In experiments, this block provides a clear gain in PESQ and delta SI-SDR with only a small increase in parameter count, and is more cost-effective than simply increasing the size of the recurrent layers.

---

## Training Objective

The model is trained to predict a CRM that improves both the STFT magnitude structure and the time-domain waveform.

- **Spectral loss**: mean squared error between compressed magnitudes  
  `|S_hat|^gamma` and `|S|^gamma` with `gamma = 0.3`  
- **Waveform loss**: L1 loss between the enhanced and clean waveforms  
- **Total loss**:  
  `L = alpha * L_mag + (1 - alpha) * L_wav`, with `alpha = 0.5` in the current experiments

---

## Results (VoiceBank+DEMAND)

Final model:

- **PESQ (wb)**: 3.035  
- **STOI**: 94.8 %  
- **Delta SI-SDR**: 10.68 dB  
- **Parameters**: 0.676 M  

### Ablation Summary

- **No complex convolution** (treat real and imaginary parts as separate real channels):
  - Parameters reduced to 0.411 M  
  - Small but consistent drop in PESQ and delta SI-SDR  
  - Complex convolutions help, but the gain is modest relative to their parameter cost

- **No frame-wise self-attention** (keep complex convolutions, remove attention block):
  - Parameters ≈ 0.638 M  
  - More noticeable drop in PESQ and delta SI-SDR than the “no complex convolution” variant  
  - Frame-wise attention at the bottleneck is a **more cost-effective** component in this setting

---

## Link

The Dataset used in this project (VoiceBank+DEMAND split) can be accessed here:
- [Training set (SharePoint link)](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/wzha0912_uni_sydney_edu_au/EdQKwK2HJN9OlrE5wmmp4jAB3opo4N1tCJwdKPJjmUjNnA?e=zbSl6b)
- [Test set (SharePoint link)](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/wzha0912_uni_sydney_edu_au/EfGRocbWKERKqToagpfQZnMBuzgWm0rv0pRHRgQF7zfGZA?e=a3EnQt)

- 
The demo video is available here
-[Demo video (SharePoint link)](https://unisydneyedu-my.sharepoint.com/:v:/g/personal/wzha0912_uni_sydney_edu_au/EUgTdmhkgddNsHU1FCq1CN0BcKdcRRU5KlkTUjfoOJ1_WA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=7eYZqf)
---

## Possible Extensions

If you build on this repository, some natural directions are:

- Explore lighter or partially shared complex convolutions  
- Slightly increase bottleneck GRU or attention size while keeping the model under 1 M parameters  
- Add explicit CPU latency / real-time factor (RTF) benchmarks  
- Evaluate on additional noisy speech corpora
