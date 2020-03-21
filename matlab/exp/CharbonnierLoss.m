function [delta, loss] = CharbonnierLoss(active, gt, mode)
% if ~exist('sparse','var')
%    sparse = 0;
% end

e = 10^(-6);

[r,c,cha,bz] = size(active);
if size(gt,1)~= r
    gt = imresize(gt,[r,c]);
end
% loss = zeros(2,1);

%%%%% crop boundaries
mask = zeros(r,c,cha,bz);
mask(16:end-15,16:end-15,:,:) = 1;
active = active.*mask;
gt = gt.*mask;

dt = active - gt;

dt2 = sqrt(dt.^2+e^2);

loss = sum(dt2(:)) / bz;


if strcmp(mode, 'train')
    delta = single(dt./dt2/bz);
else
    delta = 0;
end
end
