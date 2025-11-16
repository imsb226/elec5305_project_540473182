## Model Overview

- Sample rate: 16 kHz  
- STFT: n_fft = 512, hop = 256, Hann window  
- Input: noisy complex STFT Y(f, k), split into two channels (real part and imaginary part)  
- Output: a 2-channel complex ratio mask (CRM), [Re(M), Im(M)]  
- Enhanced spectrum: `S_hat(f, k) = M(f, k) * Y(f, k)`  
- Waveform reconstruction: inverse STFT (iSTFT) applied to `S_hat`  

### Encoder–Decoder (Complex U-Net)

The encoder–decoder follows a U-Net style architecture that downsamples only along the frequency axis and keeps the original temporal resolution. The encoder has three complex convolution blocks; each block consists of:

- complex Conv2d on the stacked real/imag channels,  
- complex BatchNorm (separate normalisation for real and imaginary parts),  
- complex PReLU activation.

Frequency is downsampled by a stride of 2 in the encoder, and upsampled by complex transposed convolution in the decoder. Three symmetric decoder blocks mirror the encoder, and skip connections link encoder and decoder features at the same scale so that fine spectral details are preserved. Real and imaginary parts are always processed jointly, rather than being treated as two independent real-valued channels.

### Dual-Path Recurrent Bottleneck (DPCRN-style)

Between the encoder and decoder, the model uses a dual-path recurrent bottleneck inspired by DPCRN/DPRNN. The bottleneck feature has shape `[B, Cg, T, F]`, where:

- `B` = batch size,  
- `Cg` = channel dimension after stacking real and imaginary parts,  
- `T` = number of time frames,  
- `F` = downsampled frequency bins.

The dual-path block contains:

- an **intra-frequency BiGRU** that runs over the frequency axis for each time frame, followed by a linear projection, LayerNorm and residual connection;  
- an **inter-time GRU** that runs over the time axis for each frequency bin, again followed by projection, LayerNorm and residual.

This structure allows the model to capture long-range dependencies in both frequency (formants, harmonics) and time (onsets, long noise segments) with relatively small hidden sizes, which is important for keeping the model lightweight.

### Frame-wise Self-Attention (Bottleneck)

On top of the dual-path bottleneck, a single **frame-wise multi-head self-attention block** is applied at the bottleneck. Each time frame is treated as a token; the frequency dimension and channels are folded into the feature dimension. Attention is computed over the time axis only, so the complexity scales roughly as O(T^2) instead of O((T * F)^2).

Implementation details:

- 1x1 convolutions generate Q, K and V from the bottleneck feature.  
- Q, K and V are reshaped into `[num_heads, T, feature_per_head]`.  
- Standard multi-head self-attention is computed over frames.  
- The result is projected back with a 1x1 convolution, followed by LayerNorm and a residual connection.

This block adds global temporal context at the bottleneck with a very small increase in parameter count, and in experiments it gives a clear gain in PESQ and delta SI-SDR.

### Training Objective and Metrics

The model is trained to predict a CRM that improves both the STFT magnitude structure and the time-domain waveform:

- spectral loss: mean squared error between compressed magnitudes `|S_hat|^gamma` and `|S|^gamma` with `gamma = 0.3`;  
- waveform loss: L1 loss between the enhanced and clean waveforms;  
- total loss: weighted sum of spectral and waveform losses.

Evaluation is done on the VoiceBank+DEMAND test set using:

- PESQ (wideband),  
- STOI,  
- delta SI-SDR (improvement over the noisy mixture).

### Test Set

The test set used in this project (VoiceBank+DEMAND split) is available here:

- [Test set (SharePoint link)](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/wzha0912_uni_sydney_edu_au/EfGRocbWKERKqToagpfQZnMBuzgWm0rv0pRHRgQF7zfGZA?e=a3EnQt)
