function playp2(img1, img2)
%PLAYP Summary of this

[h, w, c] = size(img1);
for i = 1:c
    figure(11);
    subplot(121); imagesc(img1(:,:,i)); colormap gray; axis off; axis image; title(i);
    subplot(122); imagesc(img2(:,:,i)); colormap gray; axis off; axis image; title(i);
    pause();
end
end
