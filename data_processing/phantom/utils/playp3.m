function playp3(img1, img2, img3)
%PLAYP Summary of this

if length(size(img1)) == 3
    [h, w, c] = size(img1);
    for i = 1:c
        figure(2);
        subplot(131); imagesc(img1(:,:,i)); colormap gray; axis off; axis image; title(i);
        subplot(132); imagesc(img2(:,:,i)); colormap gray; axis off; axis image; title(i);
        subplot(133); imagesc(img3(:,:,i)); colormap gray; axis off; axis image; title(i);
        pause();
    end
elseif length(size(img1)) == 2
    figure(13);
    subplot(131); imagesc(img1); colormap gray; axis off; axis image;
    subplot(132); imagesc(img2); colormap gray; axis off; axis image;
    subplot(133); imagesc(img3); colormap gray; axis off; axis image;
    pause();
end
end