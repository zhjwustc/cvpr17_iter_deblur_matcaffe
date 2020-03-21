function [deltaGrad, deltaLambda] = deconv_spat_grad_bp2(blur, kernelF, grad, deltaDeblur, lambda)
    %%%%% update grad and lambda
    %%%%% gradxF and gradyF are global parameters which are defined in 'train'
    %%%%% also consider the delta of lambda
    global gradxF;
    global gradyF;
    
    denom = 1 ./ (lambda*kernelF.*conj(kernelF) +...
        gradxF.*conj(gradxF) +...
        gradyF.*conj(gradyF) + 0.0001);
    
    deltaXGrad = ifft2(deltaDeblur);
    deltaYGrad = deltaXGrad;
    
    deltaXGrad = deltaXGrad .* conj(gradxF) .* denom;
    deltaYGrad = deltaYGrad .* conj(gradyF) .* denom;
    
    deltaXGrad = real(fft2(deltaXGrad));
    deltaYGrad = real(fft2(deltaYGrad));
    
    deltaGrad = cat(3, deltaXGrad, deltaYGrad);
    
    C = conj(kernelF) .* fft2(blur);
    D = conj(gradxF) .* fft2(grad(:,:,1,:)) + conj(gradyF) .* fft2(grad(:,:,2,:));
    E = conj(kernelF) .* kernelF;
    F = conj(gradxF) .* gradxF + conj(gradyF) .* gradyF + 0.0001;
    deltaLambda = (C.*F - D.*E) ./ (lambda*E + F).^2;
    deltaLambda = real(ifft2(deltaLambda));
    deltaLambda = deltaDeblur .* deltaLambda;
    deltaLambda = sum(deltaLambda(:));
end
