# ELEC5305_Project

Project Overview: The purpose of this project is to propose a feature fusion method to more effectively extract information from 1D audio signals and spectrograms, thereby improving the effect of speech enhancement.

Background and Motivation: Deep learning methods have achieved remarkable results in current noise removal and speech enhancement tasks. Many studies have used time-domain signals and spectrograms as input. However, methods that use both as input are less common. Therefore, this project proposes a method that fuses time-domain signal and spectrogram features for speech enhancement. This method allows the network to capture time-frequency information while retaining phase information, and encodes audio from multiple dimensions. In addition, considering that the model can respond to input quickly to meet industrial needs, this project does not intend to use time-consuming networks like Transformer, although studies have shown that such models perform well in speech enhancement tasks.

Proposed Methodology: 
  Use Python for experimentation.
  Time-domain information: models such as 1D UNet, 1D convolution, and LSTM.
  Time–frequency information: models such as 2D UNet, 2D convolution, and CNN.
  Try generative models such as GAN to directly generate more realistic speech signals, thereby avoiding the loss of speech details in regression     tasks.

Dataset: (VOICEBANK+DEMAND)
Training set: 28 subjects, total speech duration ≈ 9.4 hours, sampling frequency 48 kHz; noise SNRs: 0, 5, 10, 15 dB.
Test set: 2 unseen subjects, total speech duration ≈ 0.6 hours; noise SNRs: 2.5, 7.5, 12.5, 17.5 dB.
