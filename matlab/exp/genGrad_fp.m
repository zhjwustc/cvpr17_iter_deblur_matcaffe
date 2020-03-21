function [dx, dy] = genGrad_fp(im)
    %%%%% gradxF and gradyF are global parameters which are defined in 'train'
    global gradxF;
    global gradyF;
    
    dx = fft2(im) .* gradxF;
    dx = real(ifft2(dx));
    dy = fft2(im) .* gradyF;
    dy = real(ifft2(dy));
end