# LLIE

#Abstract</br>

Low light image enhancement is the focus of low level vision task, which restores low light image to normal light image. In recent years, image enhancement methods based on convolution neural networks are dominant. However, the Vision Transformers, which have recently made breakthroughs in advanced visual tasks, do not bring a new dimension to low light image enhancement. Therefore, CrossUFormer algorithm is proposed in this paper. Both encoder and decoder parts are composed of multiple Enhanceformer blocks. Each Enhanceformer block is composed of long and short dual-branch structures to aggregate spatial-wise information. The long branch is composed of multi-head self-attention to extract low-frequency information, while the short branch is composed of parallel convolution layer to extract high-frequency information. In order to better model the global context information, Multi-Scale Feature Cross Attention block is proposed to guide the fused multi-scale channel-wise information by cross attention to effectively connect to the decoder features. The selective fusion of encoder and decoder features are guided by Selective Kernel Feature Fusion block. Our method outperforms state-of-the-art methods on multiple datasets.

#Reference & Acknowledgement</br>

This project is released under the MIT license. The codes are built based on the excellent [TransUNet](https://github.com/Beckschen/TransUNet.git) and [DehazeFormer](https://github.com/IDKiro/DehazeFormer.git).