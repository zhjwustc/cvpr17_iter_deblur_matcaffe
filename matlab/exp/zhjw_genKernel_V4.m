function [kernel, isOutBound] = zhjw_genKernel_V4(windowSize, kernelSize, isCenter, kernelSamplePointNum, kernelValMean, kernelValVar)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate kernel as [1]. the kernel is normalized.
% the kernel is not normalized in the middle.
% 
% input:
% windowSize: size of the window (is larger than kernelSize)
% kernelSize: size of the kernel
% isCenter: whether the kernel is center in the middle. 1: centered, 0: not centered
% kernelSamplePointNum: number of sampled points to generate kernel (default as 6)
% kernelValMean: mean value of kernel (before normalization) (default as 1)
% kernelValVar: mean value of kernel (before normalization) (default as 0.5)
% output:
% kernel: generated kernel
% isOutBound: whether the kernel is out of the kernel boundary after put the center of the kernel in the middle
% reference:
% [1] Chakrabarti, A. (2016). A Neural Approach to Blind Motion Deblurring. arXiv preprint arXiv:1603.04771.
% [2] Fergus, R., Singh, B., Hertzmann, A., Roweis, S. T., & Freeman, W. T. (2006). Removing camera shake from a single photograph. ACM Transactions on Graphics (TOG), 25(3), 787-794.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(nargin==3)
    kernelSamplePointNum = 6;
    kernelValMean = 1;
    kernelValVar = 0.5;
end

if(kernelSize>windowSize)
    error('kernel size should be smaller than window size');
end

isOutBound = false;

% kernel = zeros(kernelSize);
kernel = zeros(windowSize);
sample = rand(kernelSamplePointNum, 2)*kernelSize;

order = 3;
kernelSampleInd=DEBOOR(linspace(0,1,kernelSamplePointNum-order+2),sample,linspace(0,1,1000),order);
kernelSampleInd = min(max(kernelSampleInd,1),kernelSize);

kernel(sub2ind(size(kernel),ceil(kernelSampleInd(:,2)),ceil(kernelSampleInd(:,1)))) = randn(size(kernelSampleInd,1),1)*sqrt(kernelValVar) + kernelValMean;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kernel(sub2ind(size(kernel),ceil(kernelSampleInd(1:20,2)),ceil(kernelSampleInd(1:20,1)))) = randn(20,1)*sqrt(kernelValVar) + kernelValMean*rand(1)*10;
% kernel(sub2ind(size(kernel),ceil(kernelSampleInd(end-19:end,2)),ceil(kernelSampleInd(end-19:end,1)))) = randn(20,1)*sqrt(kernelValVar) + kernelValMean*rand(1)*10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kernel(kernel<0) = 0;
if(sum(kernel(:))<=0)
    kernel(ceil(size(kernel,1)/2),ceil(size(kernel,2)/2)) = 1;
end
kernel = kernel/sum(kernel(:));
if(isCenter)    % this code is from function move_level of [2]
    meanx = sum([1:size(kernel,2)] .* sum(kernel,1));
    meany = sum([1:size(kernel,1)] .* sum(kernel,2)');
    
    offset_x = round( floor(size(kernel,2)/2)+1 - meanx );
    offset_y = round( floor(size(kernel,1)/2)+1 - meany );
%     fprintf('%f\t%f\n', abs(offset_y*2)+1, abs(offset_x*2)+1);
    
    shift_kernel = zeros(abs(offset_y*2)+1,abs(offset_x*2)+1);
    shift_kernel(abs(offset_y)+1+offset_y,abs(offset_x)+1+offset_x) = 1;
    
    kernel = conv2(kernel,shift_kernel,'same');
end
% if(abs(sum(kernel(:))-1)>0.01)
%     isOutBound = true;
%     warning('some parts of the kernel is outside the boundary after putting the center of the kernel in the middle')
% end
kernel = kernel/sum(kernel(:));




function val = DEBOOR(T,p,y,order)

% function val = DEBOOR(T,p,y,order)
%
% INPUT:  T     St�tzstellen
%         p     Kontrollpunkte (nx2-Matrix)
%         y     Auswertungspunkte (Spaltenvektor)
%         order Spline-Ordnung
%
% OUTPUT: val   Werte des B-Splines an y (mx2-Matrix)
%
% Date:   2007-11-27
% Author: Jonas Ballani
% p
m = size(p,1);
n = length(y);
X = zeros(order,order);
Y = zeros(order,order);
a = T(1);
b = T(end);
T = [ones(1,order-1)*a,T,ones(1,order-1)*b];


for l = 1:n
    t0 = y(l);
    id = find(t0 >= T);
    k = id(end);
		if (k > m)
			return;
		end
    X(:,1) = p(k-order+1:k,1);
    Y(:,1) = p(k-order+1:k,2);

    for i = 2:order
        for j = i:order
            num = t0-T(k-order+j);
            if num == 0
                weight = 0;
            else
								s = T(k+j-i+1)-T(k-order+j);
                weight = num/s;
            end
            X(j,i) = (1-weight)*X(j-1,i-1) + weight*X(j,i-1);
            Y(j,i) = (1-weight)*Y(j-1,i-1) + weight*Y(j,i-1);
        end
    end
    val(l,1) = X(order,order);
    val(l,2) = Y(order,order);
end