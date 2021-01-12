function stacked_multicoil_im = multicoil_stack(multicoil_im)
% size(multicoil_im) = h * w * coil
% make sure (abs) is done before input
% No need to make complex valued arrays three times for visualization
[h, w, c] = size(multicoil_im);
% should be h * w * 32
if c < 32
    margin = 32 - c;
    pad_multicoil_im = zeros(h, w, 32);
    pad_multicoil_im(:,:,1:end - margin) = multicoil_im;
else
    pad_multicoil_im = multicoil_im;
end

stacked_multicoil_im = zeros(h * 2, w * 16);

for i = 1:32
    h_idx = floor((i-1)/16);
    w_idx = mod(i-1, 16);
    h_val = h_idx * h + 1;
    w_val = w_idx * w + 1;
    stacked_multicoil_im(h_val:h_val + (h-1),w_val:w_val + (w-1)) = pad_multicoil_im(:,:,i);
end
end

