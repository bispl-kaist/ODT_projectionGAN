function img = denormalize_im(norm_img, maxv, minv)

img = (norm_img * maxv + minv);
end