function [delta, loss] = L2Loss(active, gt, mode)
% if ~exist('sparse','var')
%    sparse = 0;
% end
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
loss = sum(dt(:).^2) / 2 / bz;


if strcmp(mode, 'train')
    delta = single(dt/bz);
else
    delta = 0;
end
end
