Speech Enhancement (Lightweight, CPU-Friendly)



Single-channel speech enhancement targeting laptop CPU with < 1M params while maintaining practical perceptual quality. The model estimates a Complex Ratio Mask (CRM) and reconstructs enhanced waveforms via iSTFT.

TL;DR

Task: Single-channel speech enhancement (CRM, complex STFT)

Targets: PESQ(wb) ≥ 2.95, STOI ≥ 0.94, ΔSI-SDR ≥ 6.0 dB

Constraint: Params < 1M, CPU-friendly inference

Dataset: VoiceBank-DEMAND (16 kHz; train: 28 spk; test: 2 spk; SNR as in the standard split)

Backbone: U-Net-style encoder/decoder + G-DPRNN bottleneck

Head: 1×1 Conv → 2-ch CRM (real, imag) with tanh

Features

Encoder/decoder downsamples only along frequency (F) to keep temporal resolution

G-DPRNN bottleneck:

Intra-frame BiGRU over frequency bins → projection + residual

Inter-frame UniGRU over time frames → projection + residual

LayerNorm after each path

Clean skip connections, output alignment safeguards, compact parameterization

Planned: lightweight Transformer encoder, multi-objective loss (waveform + complex), structured/channel pruning
