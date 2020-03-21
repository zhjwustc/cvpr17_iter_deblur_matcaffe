clear
warning off
addpath('../');

% model path for three iterations
model_path1 = 'model1/';
model_path2 = 'model2/';
model_path3 = 'model3/';

caffe.reset_all()

Solver1 = modelconfig(model_path1);
Solver2 = modelconfig(model_path2);
Solver3 = modelconfig(model_path3);

Solver1 = dataconfig(Solver1);
Solver2 = dataconfig(Solver2);
Solver3 = dataconfig(Solver3);

%%%%% noise
Solver1.noiseLevelMin = 0.005;
Solver1.noiseLevelMax = 0.015;
%%%%% X% use salt and pepper noise
Solver1.SPNoiseP = 0.0;	%%%%%%%%%%%%%%
%%%%% salt and pepper noise
Solver1.SPnoiseLevelMin = 0;
Solver1.SPnoiseLevelMax = 0.01;
%%%%% X% use noise kernel
Solver1.NoiseKernelP = 0.8;	%%%%%%%%%%%%%%
%%%%% add kernel blur and noise
Solver1.kernelBlurVarMax = 0.9;
Solver1.kernelBlurVarMin = 0.0;
Solver1.kernelNoiseLevelMax = 0.003;
Solver1.kernelNoiseLevelMin = 0.001;
%%%%% X% use saturated
Solver1.saturateP = 0.9;
%%%%% saturated
Solver1.saturateMultiMax = 2.0;
Solver1.saturateMultiMin = 1.3;

%%%% TV weight parameter
Solver1.lambda = 0.00;

%%%%% lambda for deconv
if(~exist('lambdaAdamSolver.mat'))
    lambdaAdamSolver(1).lambda = 500;
    lambdaAdamSolver(1).iter = 1;
    lambdaAdamSolver(1).lr = 0.1;
    lambdaAdamSolver(1).beta1 = 0.9;
    lambdaAdamSolver(1).beta2 = 0.99;
    lambdaAdamSolver(1).m = 0;
    lambdaAdamSolver(1).v = 0;
    lambdaAdamSolver(1).e = 10^(-8);
    
    lambdaAdamSolver(2).lambda = 63;
    lambdaAdamSolver(2).iter = 1;
    lambdaAdamSolver(2).lr = 0.01;
    lambdaAdamSolver(2).beta1 = 0.9;
    lambdaAdamSolver(2).beta2 = 0.99;
    lambdaAdamSolver(2).m = 0;
    lambdaAdamSolver(2).v = 0;
    lambdaAdamSolver(2).e = 10^(-8);
    
    lambdaAdamSolver(3).lambda = 7.9;
    lambdaAdamSolver(3).iter = 1;
    lambdaAdamSolver(3).lr = 0.001;
    lambdaAdamSolver(3).beta1 = 0.9;
    lambdaAdamSolver(3).beta2 = 0.99;
    lambdaAdamSolver(3).m = 0;
    lambdaAdamSolver(3).v = 0;
    lambdaAdamSolver(3).e = 10^(-8);
    
    lambdaAdamSolver(4).lambda = 1;
    lambdaAdamSolver(4).iter = 1;
    lambdaAdamSolver(4).lr = 0.0001;
    lambdaAdamSolver(4).beta1 = 0.9;
    lambdaAdamSolver(4).beta2 = 0.99;
    lambdaAdamSolver(4).m = 0;
    lambdaAdamSolver(4).v = 0;
    lambdaAdamSolver(4).e = 10^(-8);
else
    load('lambdaAdamSolver.mat')
end

if(~exist('gradxF'))
    global gradxF;
    gradxF = psf2otf([0,0,0;0,1,-1;0,0,0], [Solver1.patchsize,Solver1.patchsize]);
    gradxF = repmat(gradxF, 1, 1, 1, Solver1.batchsize);
end
if(~exist('gradyF'))
    global gradyF;
    gradyF = psf2otf([0,0,0;0,1,0;0,-1,0], [Solver1.patchsize,Solver1.patchsize]);
    gradyF = repmat(gradyF, 1, 1, 1, Solver1.batchsize);
end


if isfield(Solver1, 'iter')
    begin = Solver1.iter+1;
else
    begin = 1;
    Solver1 = usePreTrainedModel(Solver1);
    Solver2 = usePreTrainedModel2(Solver2);
    Solver3 = usePreTrainedModel3(Solver3);
end

for iter = begin:Solver1.max_iter
    Solver1.iter = iter;
    Solver1.Solver_.set_iter(iter);
    Solver2.iter = iter;
    Solver2.Solver_.set_iter(iter);
    Solver3.iter = iter;
    Solver3.Solver_.set_iter(iter);
    
    [blur, clean, kernelF] = Gen_training_deblur( Solver1);
    [cleanDx, cleanDy] = genGrad_fp(clean);
    
    deblur1 = deconv_spat_grad_fp(blur, kernelF, zeros(Solver1.patchsize,Solver1.patchsize,2,Solver1.batchsize), lambdaAdamSolver(1).lambda);
    
    [deblurDx1, deblurDy1] = genGrad_fp(deblur1);
    
    batchc = {single(deblurDx1), single(permute(deblurDy1,[2,1,3,4]))}; %% remember to transpose dy!!!!
    activec = Solver1.Solver_.net.forward(batchc);
    active(:,:,1,:) = activec{1};
    active(:,:,2,:) = permute(activec{2}, [2,1,3,4]); %% remember to transpose dy!!!!
    
    denoiseGrad1 = active;

    deblur2 = deconv_spat_grad_fp(blur, kernelF, active, lambdaAdamSolver(2).lambda);
    
    [deblurDx2, deblurDy2] = genGrad_fp(deblur2);
    
    batchc = {single(deblurDx2), single(permute(deblurDy2,[2,1,3,4]))}; %% remember to transpose dy!!!!
    activec = Solver2.Solver_.net.forward(batchc);
    active(:,:,1,:) = activec{1};
    active(:,:,2,:) = permute(activec{2}, [2,1,3,4]); %% remember to transpose dy!!!!
    
    denoiseGrad2 = active;

    deblur3 = deconv_spat_grad_fp(blur, kernelF, active, lambdaAdamSolver(3).lambda);
    
    [deblurDx3, deblurDy3] = genGrad_fp(deblur3);
    
    batchc = {single(deblurDx3), single(permute(deblurDy3,[2,1,3,4]))}; %% remember to transpose dy!!!!
    activec = Solver3.Solver_.net.forward(batchc);
    active(:,:,1,:) = activec{1};
    active(:,:,2,:) = permute(activec{2}, [2,1,3,4]); %% remember to transpose dy!!!!
    
    denoiseGrad3 = active;

    deblur4 = deconv_spat_grad_fp(blur, kernelF, active, lambdaAdamSolver(4).lambda);
    

%     [deltaL1, lossL1] = CharbonnierLoss(deblur4, clean, 'train');    %% also add cropping in Charbonnier now!!!!
    [deltaL2, lossL2] = L2Loss(deblur4, clean, 'train');   
    [deltaTV, lossTV] = CharbonnierTVReg(deblur4, 'train');    %% also add cropping in Charbonnier now!!!!
    

    Solver3.lossL2(iter) = lossL2(1);
    Solver3.lossTV(iter) = lossTV(1);
    
    figure(1); imshow([clean(:,:,1,1), deblur1(:,:,1,1), deblur2(:,:,1,1), deblur3(:,:,1,1), deblur4(:,:,1,1)]);
    
%     figure(1); imshow([cleanDx(:,:,1,1), deblurDx1(:,:,1,1), denoiseDx1(:,:,1,1), deblurDx2(:,:,1,1), denoiseDx2(:,:,1,1);...
%         cleanDy(:,:,1,1), deblurDy1(:,:,1,1), denoiseDy1(:,:,1,1), deblurDy2(:,:,1,1), denoiseDy2(:,:,1,1)], [])

    drawnow;
    
    if ~isnan(Solver3.lossL2(iter))
        
        delta = deltaL2 + Solver1.lambda*deltaTV;




        [delta, deltaLambda] = deconv_spat_grad_bp2(blur, kernelF,...
            denoiseGrad3, delta, lambdaAdamSolver(4).lambda);
        
        %%%%%%%%%%%%%%% update lambda
        lambdaAdamSolver(4).m = lambdaAdamSolver(4).beta1*lambdaAdamSolver(4).m...
            + (1-lambdaAdamSolver(4).beta1)*deltaLambda;
        lambdaAdamSolver(4).v = lambdaAdamSolver(4).beta2*lambdaAdamSolver(4).v...
            + (1-lambdaAdamSolver(4).beta2)*deltaLambda^2;
        lambdaAdamSolver(4).lambda = lambdaAdamSolver(4).lambda -...
            lambdaAdamSolver(4).lr * sqrt(1-lambdaAdamSolver(4).beta2^lambdaAdamSolver(4).iter) /... 
        (1-lambdaAdamSolver(4).beta1^lambdaAdamSolver(4).iter) *...
        lambdaAdamSolver(4).m / (sqrt(lambdaAdamSolver(4).v) + lambdaAdamSolver(4).e);
        lambdaAdamSolver(4).iter = lambdaAdamSolver(4).iter + 1;
        %%%%%%%%%%%%%%%
        
        deltac = {single(delta(:,:,1,:)), single(permute(delta(:,:,2,:), [2,1,3,4]))}; %% remember to transpose dy!!!!
        deltac = Solver3.Solver_.net.backward(deltac);
        Solver3.Solver_.update();
        delta(:,:,1,:) = deltac{1};
        delta(:,:,2,:) = permute(deltac{2}, [2,1,3,4]); %% remember to transpose dy!!!!
        
        delta = genGrad_bp(delta);
        
        [delta, deltaLambda] = deconv_spat_grad_bp2(blur, kernelF,...
            denoiseGrad2, delta, lambdaAdamSolver(3).lambda);
        
        %%%%%%%%%%%%%%% update lambda
        lambdaAdamSolver(3).m = lambdaAdamSolver(3).beta1*lambdaAdamSolver(3).m...
            + (1-lambdaAdamSolver(3).beta1)*deltaLambda;
        lambdaAdamSolver(3).v = lambdaAdamSolver(3).beta2*lambdaAdamSolver(3).v...
            + (1-lambdaAdamSolver(3).beta2)*deltaLambda^2;
        lambdaAdamSolver(3).lambda = lambdaAdamSolver(3).lambda -...
            lambdaAdamSolver(3).lr * sqrt(1-lambdaAdamSolver(3).beta2^lambdaAdamSolver(3).iter) /... 
        (1-lambdaAdamSolver(3).beta1^lambdaAdamSolver(3).iter) *...
        lambdaAdamSolver(3).m / (sqrt(lambdaAdamSolver(3).v) + lambdaAdamSolver(3).e);
        lambdaAdamSolver(3).iter = lambdaAdamSolver(3).iter + 1;
        %%%%%%%%%%%%%%%

        deltac = {single(delta(:,:,1,:)), single(permute(delta(:,:,2,:), [2,1,3,4]))}; %% remember to transpose dy!!!!
        deltac = Solver2.Solver_.net.backward(deltac);
        Solver2.Solver_.update();
        delta(:,:,1,:) = deltac{1};
        delta(:,:,2,:) = permute(deltac{2}, [2,1,3,4]); %% remember to transpose dy!!!!
        
        delta = genGrad_bp(delta);
        
        [delta, deltaLambda] = deconv_spat_grad_bp2(blur, kernelF,...
            denoiseGrad1, delta, lambdaAdamSolver(2).lambda);
        
        %%%%%%%%%%%%%%% update lambda
        lambdaAdamSolver(2).m = lambdaAdamSolver(2).beta1*lambdaAdamSolver(2).m...
            + (1-lambdaAdamSolver(2).beta1)*deltaLambda;
        lambdaAdamSolver(2).v = lambdaAdamSolver(2).beta2*lambdaAdamSolver(2).v...
            + (1-lambdaAdamSolver(2).beta2)*deltaLambda^2;
        lambdaAdamSolver(2).lambda = lambdaAdamSolver(2).lambda -...
            lambdaAdamSolver(2).lr * sqrt(1-lambdaAdamSolver(2).beta2^lambdaAdamSolver(2).iter) /... 
        (1-lambdaAdamSolver(2).beta1^lambdaAdamSolver(2).iter) *...
        lambdaAdamSolver(2).m / (sqrt(lambdaAdamSolver(2).v) + lambdaAdamSolver(2).e);
        lambdaAdamSolver(2).iter = lambdaAdamSolver(2).iter + 1;
        %%%%%%%%%%%%%%%
        
        deltac = {single(delta(:,:,1,:)), single(permute(delta(:,:,2,:), [2,1,3,4]))}; %% remember to transpose dy!!!!
        deltac = Solver1.Solver_.net.backward(deltac);
        Solver1.Solver_.update();
        delta(:,:,1,:) = deltac{1};
        delta(:,:,2,:) = permute(deltac{2}, [2,1,3,4]); %% remember to transpose dy!!!!
        
        delta = genGrad_bp(delta);
        
        [delta, deltaLambda] = deconv_spat_grad_bp2(blur, kernelF,...
            zeros(Solver1.patchsize,Solver1.patchsize,2,Solver1.batchsize), delta, lambdaAdamSolver(1).lambda);
        
        %%%%%%%%%%%%%%% update lambda
        lambdaAdamSolver(1).m = lambdaAdamSolver(1).beta1*lambdaAdamSolver(1).m...
            + (1-lambdaAdamSolver(1).beta1)*deltaLambda;
        lambdaAdamSolver(1).v = lambdaAdamSolver(1).beta2*lambdaAdamSolver(1).v...
            + (1-lambdaAdamSolver(1).beta2)*deltaLambda^2;
        lambdaAdamSolver(1).lambda = lambdaAdamSolver(1).lambda -...
            lambdaAdamSolver(1).lr * sqrt(1-lambdaAdamSolver(1).beta2^lambdaAdamSolver(1).iter) /... 
        (1-lambdaAdamSolver(1).beta1^lambdaAdamSolver(1).iter) *...
        lambdaAdamSolver(1).m / (sqrt(lambdaAdamSolver(1).v) + lambdaAdamSolver(1).e);
        lambdaAdamSolver(1).iter = lambdaAdamSolver(1).iter + 1;
        %%%%%%%%%%%%%%%
        
        fprintf('lambda:\t%05f\t%05f\t%05f\t%05f\n', lambdaAdamSolver(1).lambda, lambdaAdamSolver(2).lambda, lambdaAdamSolver(3).lambda, lambdaAdamSolver(4).lambda);
        
    else
        error('Model NAN.')
    end
    
    % vis
    if ~mod(iter,10)
        fprintf('========Processed iter %.6d, ',iter);
        fprintf('loss_L2: %d=======', mean(Solver3.lossL2(iter-9:iter)));
        fprintf('loss_TV: %d=======', mean(Solver3.lossTV(iter-9:iter)));
        fprintf('\n');
    end    
    
    if ~mod(iter,1000)
	outimg = [deblur1(:,:,:,1), deblur2(:,:,:,1), deblur3(:,:,:,1), deblur4(:,:,:,1), clean(:,:,:,1)];
	imwrite(outimg, ['./out/', num2str(iter), '.png']);
        Solver1.Solver_.save();
        Solver2.Solver_.save();
        Solver3.Solver_.save();
        % save mat
	if mod(iter-2000,10000)
            delete(['./model1/LRNN_iter_' num2str(iter-2000) '.caffemodel'])
            delete(['./model1/LRNN_iter_' num2str(iter-2000) '.solverstate'])
            delete(['./model2/LRNN_iter_' num2str(iter-2000) '.caffemodel'])
            delete(['./model2/LRNN_iter_' num2str(iter-2000) '.solverstate'])
            delete(['./model3/LRNN_iter_' num2str(iter-2000) '.caffemodel'])
            delete(['./model3/LRNN_iter_' num2str(iter-2000) '.solverstate'])
	end
        Solver = Solver1;
        save(Solver1.matfile, 'Solver');
        Solver = Solver2;
        save(Solver2.matfile, 'Solver');
        Solver = Solver3;
        save(Solver3.matfile, 'Solver');
        
        save('lambdaAdamSolver', 'lambdaAdamSolver')
    end
    
    
    
    
    
end
