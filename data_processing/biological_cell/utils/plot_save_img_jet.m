function plot_save_img_jet(img, fname, maxv, minv)

    minv = 1.34;
    % Plots, and then saves a single image into '.png' extension wout margin

    hdl_fig = figure('Position', [0, 0, 500, 500]);
    set(gca, 'position', [0, 0, 1, 1]);
    set(gcf, 'paperpositionmode', 'auto');

    figure(hdl_fig); colormap jet; axis image off; colorbar;
    imagesc(img, [minv maxv]);

    % r300 refers to 300dpi
    print(fname, '-dpng', '-r300'); 
    close all;
end