function playp(img)
%PLAYP Summary of this function goes here
%   Detailed explanation goes here
figure(12);

[h, w, d, c] = size(img);
for i = 1:d
    figure(12);
    slice = squeeze(img(:, :, i, :));
    slice = slice ./ max(slice(:));
    imshow(slice); title(i);
    pause();
end
end