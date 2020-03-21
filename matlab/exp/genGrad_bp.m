function [deltaIm] = genGrad_bp(deltaGrad)
    %%%%% gradxF and gradyF are global parameters which are defined in 'train'
    global gradxF;
    global gradyF;
    
    deltaIm = real(fft2(ifft2(deltaGrad(:,:,1,:)).*gradxF)) + real(fft2(ifft2(deltaGrad(:,:,2,:)).*gradyF));
end