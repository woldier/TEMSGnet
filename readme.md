## Semi-Airborne Transient Electromagnetic Denoising Through Variation Diffusion Model

<b>
Bin Wang, 
<a href='https://dengfei-ailab.github.io'>Deng Fei</a>, 
<a href='https://github.com/jiangpeifan'>Peifan Jiang</a>
<a href=''>Xuben Wang</a>
<a href=''> Ming Guo</a>
</b>

<hr>
<i>		In geophysical exploration methods, the Semi-Airborne Transient Electromagnetic (SATEM) technique offers substantial exploration depth and the capability to acquire precise underground characteristics.  However, the SATEM signal received by the induction coil has a low amplitude in the middle and late stages and is susceptible to various noise effects. 
			Although a variety of conventional denoising methods are available today, their performance is often unsatisfactory, necessitating manual adjustments of the denoising outcomes. 
			With the emergence of deep learning in various fields, several SATEM denoising neural network models have been proposed.  However, these models are discriminative model and focus on modeling conditional probabilities, with insufficient generalization capabilities when faced with small data sets. On the other hand, the Variational Diffusion Model (VDM) is a generative model that captures the joint distribution and exhibits excellent generalization capabilities. 
			Building upon this premise, we explored a VDM-based denoising approach for SATEM. Nevertheless, due to the uncontrollable nature of the results generated by VDM, it is not possible to use VDM directly for denoising SATEM signals. To address this issue, we introduce the VDM-based denoiser with constraints, which incorporates guiding conditions to constrain the model and generate desired denoising results,  and proposed a new SATEM signal denoising network called TEMSGnet. To further enhance the practical applicability of our method in real-world scenarios, we incorporate a wavelet transform-based supervised fine-tuning strategy. Experimental results demonstrate that TEMSGnet achieves effective denoising performance on both synthetic and real datasets. Furthermore, through inversion, TEMSGnet can more accurately approximate the true subsurface characteristics, thereby effectively reflecting the denoised outcomes.</i>

---

## Package dependencies
The project is built with `PyTorch 1.13.1`, `Python3.7`, `CUDA11.7`.
