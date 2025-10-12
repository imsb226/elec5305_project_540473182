Speech Enhancement (Lightweight, CPU-Friendly)

Single-channel speech enhancement (WIP). Targets laptop-CPU deployment under < 1M parameters while maintaining practical perceptual/intelligibility quality. Input is complex STFT; the model predicts a Complex Ratio Mask (CRM) and reconstructs waveforms via iSTFT. Backbone: U-Net encoder/decoder that downsamples only along the frequency axis, with a G-DPRNN bottleneck (intra-frame BiGRU, inter-frame UniGRU, both with projection, residuals, and LayerNorm).

Status

WIP: interfaces and experiments are evolving; breaking changes may occur.

Goals

PESQ(wb) >= 2.95

STOI >= 0.94

Delta SI-SDR >= 6.0 dB

Parameters < 1.0M and CPU-friendly latency

Dataset and Protocol

VoiceBank-DEMAND (16 kHz, standard split and SNR settings)

Recommend at least 3 random seeds with mean ± std reporting

Disclose parameter count and CPU-side RTF/latency

Method Snapshot

Frequency-only downsampling to preserve temporal resolution

G-DPRNN bottleneck:

Intra-frame BiGRU over frequency -> projection + residual + LayerNorm

Inter-frame UniGRU over time -> projection + residual + LayerNorm

Head outputs a 2-channel CRM (real, imag) applied to the mixture spectrum before iSTFT

Current Progress

 Base model prototype (U-Net + G-DPRNN + CRM)

 Data preprocessing and alignment checks (16 kHz, unified STFT)

 Training logging and metrics (PESQ/STOI/Delta SI-SDR, params, RTF/latency)

 Initial baselines and 3-seed reproducibility

 Ablations (bottleneck repeats, hidden sizes, freq-only downsampling)

Roadmap

 Lightweight Transformer encoder for global context

 Multi-objective losses (time-domain MSE + complex-spectrum MSE)

 Structured/channel pruning for smaller models and lower latency

 Deeper error analysis and uncertainty reporting

 ONNX export and CPU demo (text-only quick guide)

Results (TBD)

To be added: mean ± std for PESQ/STOI/Delta SI-SDR; parameter count (M); CPU RTF/latency; failure cases and error-distribution summaries.
