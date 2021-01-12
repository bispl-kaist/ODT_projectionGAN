function norm_img = normalize_im(img)
minv = min(img(:));

img = img - minv;

maxv = max(img(:));

norm_img = img / maxv;
end