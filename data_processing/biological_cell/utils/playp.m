function playp(img)
%PLAYP Summary of this function goes here
%   Detailed explanation goes here
if length(size(img)) == 2
    figure(11);
    imagesc(abs(img)); colormap gray; axis off; axis image;
elseif length(size(img)) == 3
    [h, w, c] = size(img);
    for i = 1:c
        figure(11);
        imagesc(abs(img(:,:,i))); colormap gray; axis off; axis image; title(i);
        pause();
    end
end
end

