clear; clc;

restoredefaultpath();
addpath('./utils');
addpath('./tomo_func');

snum       = 1;

%%
input_root = ['../../dataset/phantom/' sprintf('%03d/',snum)];
recon_root = ['../../recons/phantom/recon_axi/' sprintf('%03d/', snum)];


%%
figure(18); 
for i = 1:5:360
    fname = ['p' int2str(snum) '_ang' int2str(i)];
    load([input_root fname], 'input');
    load([recon_root fname], 'recons');
    recons = squeeze(recons);

    sgtitle(i);
    subplot(121); imagesc(input); colormap gray; axis off image;
    subplot(122); imagesc(recons); colormap gray; axis off image;
    pause();
end
