function [norm_img, maxv, minv] = normalize_im_verbose(img)

maxv = max(img(:));
minv = min(img(:));

norm_img = (img - minv) / maxv;
end