function [delta, loss] = L2Loss(active, gt, mode)
% if ~exist('sparse','var')
%    sparse = 0;
% end
[r,c,cha,bz] = size(active);
if size(gt,1)~= r
    gt = imresize(gt,[r,c]);
end
% loss = zeros(2,1);

dt = active - gt;
loss = sum(abs(dt(:))) / bz;


if strcmp(mode, 'train')
    delta = single(sign(dt)/bz);
else
    delta = 0;
end
end
