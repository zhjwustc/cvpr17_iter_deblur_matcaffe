% v1 support inputs with image gradients (x and y)
% select region with large error
function [batch, gt, kF] = Gen_training_deblur( Solver)


batch = single(zeros(Solver.patchsize,Solver.patchsize,1,Solver.batchsize));
gt = single(zeros(Solver.patchsize,Solver.patchsize,1,Solver.batchsize));
kF = single(zeros(Solver.patchsize,Solver.patchsize,1,Solver.batchsize));
rng('shuffle');
idpool = randperm(Solver.data.train_num);
count = 1;
while count <= Solver.batchsize
    idx = idpool(count);
    clean = im2single(imread(Solver.data.trainCleanlst{idx}));
    blur = im2single(imread(Solver.data.trainBlurlst{idx}));
    
    if(strcmp(Solver.data.trainDataType{idx}, 'flickr'))    %% only add saturated region in flickr data
	if(rand(1)<Solver.saturateP)
            %%%%% generate saturated regions
            saturateMulti = rand(1)*(Solver.saturateMultiMax-Solver.saturateMultiMin) + Solver.saturateMultiMax;
            clean = clean*saturateMulti;
            blur = blur*saturateMulti;
            clean(clean>1) = 1;
            blur(blur>1) = 1;
	end
    end
    
    %%%%% whether use gt kernel or noise kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    load(Solver.data.trainkernelGtlst{idx})
    if(rand(1)<Solver.NoiseKernelP)
        kernelBlurKernelVar = rand(1)*(Solver.kernelBlurVarMax-Solver.kernelBlurVarMin) + Solver.kernelBlurVarMin;
        kernelBlurKernel = fspecial('Gaussian',9,kernelBlurKernelVar);
        kernelBlur = conv2(kernel,kernelBlurKernel,'same');
        kernelNoiseLevel = rand(1)*(Solver.kernelNoiseLevelMax-Solver.kernelNoiseLevelMin) + Solver.kernelNoiseLevelMin;
        kernelNoise = kernelBlur + randn(size(kernelBlur))*kernelNoiseLevel;
        kernelNoiseMax = max(kernelNoise(:));
        kernelNoise(kernelNoise<kernelNoiseMax/5) = 0;
        kernelNoise = kernelNoise/sum(kernelNoise(:));

        kernel = kernelNoise;
    end
    
    if(rand(1)<Solver.SPNoiseP)
	%%%%% add salt and pepper noise
        SPnoiseLevel = rand(1)*(Solver.SPnoiseLevelMax-Solver.SPnoiseLevelMin) + Solver.SPnoiseLevelMin;
        blur = imnoise(blur,'salt & pepper',SPnoiseLevel);
    else
	%%%%% add noise into blurry image
        noiseLevel = rand(1)*(Solver.noiseLevelMax-Solver.noiseLevelMin) + Solver.noiseLevelMin;
        blur = blur + randn(size(blur))*noiseLevel;    %%%%%%%%%%%%%%%%%%%%%%%%%%% noise for blur image
    end
    
    %%%%% padding for both clean and blur image
    clean = clean(16:end-15,16:end-15);
    clean = wrap_boundary_with_edgetaper(clean, size(clean)+31-1, ceil([31,31]/2));
    clean = circshift(clean, [(31-1)/2,(31-1)/2]);

    
    blur = blur(16:end-15,16:end-15);
    blur = wrap_boundary_with_edgetaper(blur, size(blur)+31-1, ceil([31,31]/2));
    blur = circshift(blur, [(31-1)/2,(31-1)/2]);
    
    kernelF = psf2otf(kernel, size(blur));
    batch(:,:,:,count) = single(blur);
    gt(:,:,:,count) = single(clean);  
    kF(:,:,:,count) = kernelF;   
    count = count + 1;
end
end

