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

# Early changes

This report makes only one change in the frequency domain branch: placing the frequency-dimension Transformer before the frequency-domain BiGRU, forming the structure “Freq-TF → frequency-domain BiGRU → time-domain GRU.” The rationale is that mask estimation requires leveraging non-local correlations across frequency bands, so attention first aggregates global frequency context, and then the GRU handles local continuity and boundary refinement. Since there is no causal constraint along the frequency axis and frequencies at the bottleneck are already downsampled, the computation-to-benefit ratio is better. Experimental results show that under a consistent evaluation pipeline, PESQ sees a slight improvement while STOI remains basically unchanged: this modification primarily improves cross-band gain consistency and noise perception quality (more beneficial for PESQ), whereas STOI relies more on temporal envelopes and low-frequency modulation cues, so it is less affected. Further attempts to introduce a time-dimension Transformer (with causal masking) under the same settings resulted in a decrease in PESQ, possibly due to excessive noise suppression or over-flattening of envelopes caused by time attention, limited effective context, and increased optimization difficulty. Subsequent work will prioritize exploring a combination of channel attention and small-scale GRU to enhance temporal modeling: channel attention dynamically recalibrates speech-relevant features, while TCN/GRU provides sequential modeling and local dynamic control, aiming to improve STOI without significantly increasing latency and parameters, while maintaining or improving PESQ.

# Final changes


# Results 

The final model：

PESQ: 3.029

STOI: 94.8%

Delta SI-SDR = 10.64 dB

parameter count (M): 0.676

