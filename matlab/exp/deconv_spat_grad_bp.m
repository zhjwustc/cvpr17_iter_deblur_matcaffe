function deltaGrad = deconv_spat_grad_bp(kernelF, deltaDeblur, lambda)
    %%%%% only update grad
    %%%%% gradxF and gradyF are global parameters which are defined in 'train'
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
end
