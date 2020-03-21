clear
addpath('../');
caffe.reset_all();

model_path1 = 'model1/';
model_path2 = 'model2/';
model_path3 = 'model3/';

outpath = 'out/';
blurpath = 'blur image path';
kernelpath = 'blur kernel path';

img_list = dir(fullfile(blurpath, '*.jpg'));

load('lambdaAdamSolver.mat')

% for i=1:length(img_list)
for i=1:length(img_list)
    blur = imread(fullfile(blurpath, img_list(i).name));
    blur = im2single(blur);
    
    kernel = imread(fullfile(kernelpath, [img_list(i).name(1:end-4) '_out_kernel.png']));
    kernel = im2single(kernel);
    if(size(kernel,3)==3)
        kernel = mean(kernel,3);
    end
    kernel = kernel/sum(kernel(:));
    
    [h0,w0,c] = size(blur);

    [padH, padW] = padSizeEst(h0,w0);
    
    blur = wrap_boundary_with_edgetaper(blur, size(blur)+[padH, padW, 0], 16);
    blur = circshift(blur, [15,15]);
    [h,w,c] = size(blur);
    
    kernelF = psf2otf(kernel, [h,w]);

    global gradxF;
    gradxF = psf2otf([0,0,0;0,1,-1;0,0,0], [h,w]);
    global gradyF;
    gradyF = psf2otf([0,0,0;0,1,0;0,-1,0], [h,w]);
    
    creatPrototxt(w, h, './model1/LRNN_train_CVPR17.prototxt');
    creatPrototxt(w, h, './model2/LRNN_train_CVPR17.prototxt');
    creatPrototxt(w, h, './model3/LRNN_train_CVPR17.prototxt');
    
    Solver1 = modelconfig_diffSize(model_path1);
    Solver2 = modelconfig_diffSize(model_path2);   
    Solver3 = modelconfig_diffSize(model_path3);   
    
    active = zeros(h,w,2);
    deblur = zeros(h,w,c);
    for j=1:c
        deblur1 = deconv_spat_grad_fp(blur(:,:,j), kernelF, zeros(h,w,2), lambdaAdamSolver(1).lambda);
        [deblurDx1, deblurDy1] = genGrad_fp(deblur1);
        batchc = {single(deblurDx1),single(deblurDy1')}; %% remember to transpose dy!!!!
        activec = Solver1.Solver_.net.forward(batchc);
        active(:,:,1) = activec{1};
        active(:,:,2) = activec{2}'; %% remember to transpose dy!!!!
        denoiseGrad1 = active;
        
        deblur2 = deconv_spat_grad_fp(blur(:,:,j), kernelF, active, lambdaAdamSolver(2).lambda);   
        [deblurDx2, deblurDy2] = genGrad_fp(deblur2);
        batchc = {single(deblurDx2),single(deblurDy2')}; %% remember to transpose dy!!!!   
        activec = Solver2.Solver_.net.forward(batchc);
        active(:,:,1,:) = activec{1};
        active(:,:,2,:) = activec{2}'; %% remember to transpose dy!!!!
        denoiseGrad2 = active;
        
        deblur3 = deconv_spat_grad_fp(blur(:,:,j), kernelF, active, lambdaAdamSolver(3).lambda);   
        [deblurDx3, deblurDy3] = genGrad_fp(deblur3);
        batchc = {single(deblurDx3),single(deblurDy3')}; %% remember to transpose dy!!!!   
        activec = Solver3.Solver_.net.forward(batchc);
        active(:,:,1,:) = activec{1};
        active(:,:,2,:) = activec{2}'; %% remember to transpose dy!!!!
        denoiseGrad3 = active;
        
        deblur4 = deconv_spat_grad_fp(blur(:,:,j), kernelF, active, lambdaAdamSolver(4).lambda); 
        
        deblur(:,:,j) = deblur4;
    end
    
    deblur = deblur(15+1:15+h0,15+1:15+w0,:);
    
    imwrite(deblur, fullfile(outpath, [img_list(i).name(1:end-4) '.png']))
    
    caffe.reset_all();
end