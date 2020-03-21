function [padH, padW] = padSizeEst(h,w)

%%%%%%
% make sure it can be divided by 4
% padding about 31 pixs
%%%%%%

if(mod(h,4)==0)
    padH = 32;
else if(mod(h,4)==1)
        padH = 33;
    else if(mod(h,4)==2)
            padH = 34;
        else if(mod(h,4)==3)
                padH = 31;
            end
        end
    end
end
if(mod(w,4)==0)
    padW = 32;
else if(mod(w,4)==1)
        padW = 33;
    else if(mod(w,4)==2)
            padW = 34;
        else if(mod(w,4)==3)
                padW = 31;
            end
        end
    end
end