# [NTGAN-Learning-Blind-Image-Denoising-Without-Clean-Reference](https://github.com/RickZ1010/NTGAN-Noise-Transference-Generative-Adversarial-Network-for-Blind-Unsupervised-Image-Denoising)

This is the implementation of the following paper:

**NTGAN: Learning Blind Image Denoising without Clean Reference**

*Rui Zhao, Daniel P.K. Lun, and Kin-Man Lam*

Abstract: Recent studies on learning-based image denoising have achieved promising performance on various noise reduction tasks. Most of these deep denoisers are trained either under the supervision of clean references, or unsupervised with the assumption that noise is signal independent. The signal-independent assumption on the noise leads to a poor generalization when facing real-world photographs. To address this issue, we propose a novel deep unsupervised image-denoising method by regarding the noise reduction task as a special case of the noise transference task. Learning noise transference enables the network to acquire the denoising ability by only observing the corrupted samples. The results on real-world denoising benchmarks demonstrate that our proposed method achieves state-of-the-art performance on removing realistic noises, making it a potential solution to practical noise reduction problems.

## Dependencies
Python >= 3.6.5, Pytorch >= 0.4.1, and cuda-9.2.
