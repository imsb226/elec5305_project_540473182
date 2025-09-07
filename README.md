# ELEC5305_Project

<b>Overview</b>

This project propose a feature-fusion method for speech enhancement. It combines 1D audio waveforms and time–frequency spectrograms. Using both inputs together is less common in prior work. The fusion captures local time–frequency patterns and keeps phase cues from the waveform. The model encodes speech from multiple views. To meet latency and compute limits, this project avoid heavy Transformer models, even though they can perform well.

<b>Proposed Methodology</b>
<ul> <li>Use Python for experimentation.</li> <li>Time-domain information</b>: models such as 1D UNet, 1D convolution, and LSTM.</li> <li>Time–frequency information: models such as 2D UNet, 2D convolution, and CNN.</li> <li>Try generative models such as GAN or Diffusion to directly generate more realistic speech signals, thereby avoiding the loss of speech details in regression tasks.</li> </ul>
<b>Dataset</b> (VOICEBANK+DEMAND)
<ul> <li><b>Training set</b>: 28 subjects, total speech duration ≈ 9.4 hours, sampling frequency 48 kHz; noise SNRs: 0, 5, 10, 15 dB.</li> <li><b>Test set</b>: 2 unseen subjects, total speech duration ≈ 0.6 hours; noise SNRs: 2.5, 7.5, 12.5, 17.5 dB.</li> </ul>
