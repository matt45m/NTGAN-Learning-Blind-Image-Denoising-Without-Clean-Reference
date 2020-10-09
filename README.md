# [NTGAN-Learning-Blind-Image-Denoising-Without-Clean-Reference](https://github.com/RickZ1010/NTGAN-Noise-Transference-Generative-Adversarial-Network-for-Blind-Unsupervised-Image-Denoising)

This is the implementation of the following paper:

**NTGAN: Learning Blind Image Denoising without Clean Reference**

*Rui Zhao, Daniel P.K. Lun, and Kin-Man Lam*

Abstract: Recent studies on learning-based image denoising have achieved promising performance on various noise reduction tasks. Most of these deep denoisers are trained either under the supervision of clean references, or unsupervised on synthetic noise. The assumption with the synthetic noise leads to poor generalization when facing real photographs. To address this issue, we propose a novel deep unsupervised image-denoising method by regarding the noise reduction task as a special case of the noise transference task. Learning noise transference enables the network to acquire the denoising ability by only observing the corrupted samples. The results on real-world denoising benchmarks demonstrate that our proposed method achieves state-of-the-art performance on removing realistic noises, making it a potential solution to practical noise reduction problems.

[Paper](https://www.bmvc2020-conference.com/assets/papers/0046.pdf), [Supplementary material](https://www.bmvc2020-conference.com/assets/supp/0046_supp.pdf), [Video](https://www.bmvc2020-conference.com/conference/papers/paper_0046.html)

## Dependencies
Python >= 3.6.5, Pytorch >= 0.4.1, cuda-9.2, and [colour-demosaicing](https://github.com/colour-science/colour-demosaicing)

## Framework

<div  align="center">    
<img src="https://github.com/RickZ1010/NTGAN-Learning-Blind-Image-Denoising-Without-Clean-Reference/blob/master/figs/fig1.png?raw=true" width=700/>
</div>

## Results
### Realistic noise removal
<div align=center><img width="700" src="https://github.com/RickZ1010/NTGAN-Learning-Blind-Image-Denoising-Without-Clean-Reference/blob/master/figs/table1.png?raw=true"/></div>
<div align=center><img width="800" src="https://github.com/RickZ1010/NTGAN-Learning-Blind-Image-Denoising-Without-Clean-Reference/blob/master/figs/table2_.png?raw=true"/></div>


## Citation

    @INPROCEEDINGS{ZhaoNTGAN2020, 
        author={R. {Zhao} and D. P. K. {Lun} and K. {Lam}}, 
        booktitle={2020 British Machine Vision Conference (BMVC)}, 
        title={NTGAN: Learning Blind Image Denoising without Clean Reference}, 
        year={2020}, 
        month={Sep.}
        }
