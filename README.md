This is a matcaffe version of training and testing for the image deblurring algorithm described in the paper: 
Jiawei Zhang, Jinshan Pan, Wei-Sheng Lai, Rynson W.H. Lau, Ming-Hsuan Yang, "Learning Fully Convolutional Networks for Iterative Non-blind Deconvolution", CVPR 2017. 

This implementation is not well organized. Let me know if you have problems.

1. use train.m and test.m for training and testing
2. change data path in dataconfig.m after generating the blurry and sharp image patches with blur kernels
3. padding and noise are added in Gen_training_deblur.m
4. can manually load pretrained weights in usePreTrainedModel.m, usePreTrainedModel2.m and usePreTrainedModel3.m for each iterations (can also load weights from matconvnet model by usePreTrainedModelFromMatconvnet)
