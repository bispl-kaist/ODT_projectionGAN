function save_img(img, fname)

% Normalizes, then save img into a .png extension

    img = img - min(img(:));
    img = img / max(img(:));
    
    imwrite(img, fname);
end