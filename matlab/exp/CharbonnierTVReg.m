function [delta, loss] = CharbonnierTVReg(active, mode)
% if ~exist('sparse','var')
%    sparse = 0;
% end

e = 10^(-6);

[r,c,cha,bz] = size(active);
% if size(gt,1)~= r
%     gt = imresize(gt,[r,c]);
% end
% loss = zeros(2,1);

%%%% crop boundaries
mask = zeros(r,c,cha,bz);
mask(16:end-15,16:end-15,:,:) = 1;
active = active.*mask;

x = active(:,2:end,:,:) - active(:,1:end-1,:,:);
loss = sum(sqrt((x(:).^2+e^2)));
if strcmp(mode, 'train')
    x = x./sqrt((x.^2+e^2));
    x = cat(2, zeros(r,1,cha,bz), x, zeros(r,1,cha,bz));
    x = x(:,1:end-1,:,:) - x(:,2:end,:,:);
end


y = active(2:end,:,:,:) - active(1:end-1,:,:,:);
loss = loss + sum(sqrt((y(:).^2+e^2)));
if strcmp(mode, 'train')
    y = y./sqrt((y.^2+e^2));
    y = cat(1, zeros(1,c,cha,bz), y, zeros(1,c,cha,bz));
    y = y(1:end-1,:,:,:) - y(2:end,:,:,:);    
end


delta = x+y;

end
