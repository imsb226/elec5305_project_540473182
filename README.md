## Speech Enhancement (Lightweight, CPU-Friendly)

Single-channel speech enhancement (WIP). Targets laptop-CPU deployment under < 1M parameters while maintaining practical perceptual/intelligibility quality. Input is complex STFT; the model predicts a Complex Ratio Mask (CRM) and reconstructs waveforms via iSTFT. Backbone: U-Net encoder/decoder that downsamples only along the frequency axis, with a G-DPRNN bottleneck (intra-frame BiGRU, inter-frame UniGRU, both with projection, residuals, and LayerNorm).

# Status

WIP: interfaces and experiments are evolving; breaking changes may occur.

# Goals

PESQ(wb) >= 2.95

STOI >= 0.94

Delta SI-SDR >= 6.0 dB

Parameters < 1.0M and CPU-friendly latency

# Dataset and Protocol

VoiceBank-DEMAND (16 kHz, standard split and SNR settings)

Recommend at least 3 random seeds with mean ± std reporting

Disclose parameter count and CPU-side RTF/latency

# Overview of basic model

Frequency-only downsampling to preserve temporal resolution

G-DPRNN bottleneck:

Intra-frame BiGRU over frequency -> projection + residual + LayerNorm

Inter-frame UniGRU over time -> projection + residual + LayerNorm

Head outputs a 2-channel CRM (real, imag) applied to the mixture spectrum before iSTFT

# Current Model

To better capture non-local dependencies across frequency bands for mask estimation on the time–frequency spectrum, This project places a frequency-axis Transformer before the frequency-domain GRU. This yields a small but consistent improvement over the baseline.

# Roadmap

 Lightweight Transformer encoder for global context 

 Multi-objective losses (time-domain MSE + complex-spectrum MSE)

 Structured/channel pruning for smaller models and lower latency

 Deeper error analysis and uncertainty reporting

 ONNX export and CPU demo (text-only quick guide)

# Results 
The basic model with three layers of DPRCN Blocks：\n
PESQ: 2.929
STOI: 94.4%
Si-SNR: 18.38dB
parameter count (M): 0.158434

The model after adding a frequency-dimension transformer：
PESQ: 2.954
STOI: 94.3%
Si-SNR: 18.51dB
parameter count (M): 0.184258

