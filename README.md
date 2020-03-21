1. use train.m and test.m for training and testing
2. change data path in dataconfig.m after generating the blurry and sharp image patches with blur kernels
3. padding and noise are added in Gen_training_deblur.m
4. manually load pretrained weights in usePreTrainedModel.m, usePreTrainedModel2.m and usePreTrainedModel3.m for each iterations (can also load weights from matconvnet model by usePreTrainedModelFromMatconvnet)
