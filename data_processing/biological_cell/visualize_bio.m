clear; clc;

restoredefaultpath();
addpath('./utils');
addpath('./tomo_func');


%% Load data, and apply projection

root_save = '../../dataset/bio/';
root_input_axi = [root_save 'Input_infer_axi/'];
root_input_cor = [root_save 'Input_infer_cor/'];
root_input_sag = [root_save 'Input_infer_sag/'];

root_recon = '../../recons/bio/';
root_recon_vol = '../../recons_vol/'; mkdir(root_recon_vol);
root_recon = [root_recon];
root_recon_axi = [root_recon 'recon_axi/'];
root_recon_cor = [root_recon 'recon_cor/'];
root_recon_sag = [root_recon 'recon_sag/'];

specimen_type = '20181120_NIH3T3_LipidDroplet(PM)_0.313.30/';
specimen_name = '20181120.190011.239.Default-092/';

root_input_axi_ss = [root_input_axi specimen_type specimen_name];
root_input_cor_ss = [root_input_cor specimen_type specimen_name];
root_input_sag_ss = [root_input_sag specimen_type specimen_name];

root_recon_axi_ss = [root_recon_axi specimen_type specimen_name];
root_recon_cor_ss = [root_recon_cor specimen_type specimen_name];
root_recon_sag_ss = [root_recon_sag specimen_type specimen_name];

for i = 1:360
    load([root_input_axi_ss 'a' num2str(i)], 'proj');
    axi_proj = proj;
    load([root_input_cor_ss 'a' num2str(i)], 'proj');
    cor_proj = proj;
    load([root_input_sag_ss 'a' num2str(i)], 'proj');
    sag_proj = proj;

    load([root_recon_axi_ss 'a' num2str(i)], 'recons');
    axi_recons = recons;
    load([root_recon_cor_ss 'a' num2str(i)], 'recons');
    cor_recons = recons;
    load([root_recon_sag_ss 'a' num2str(i)], 'recons');
    sag_recons = recons;

    figure(1);
    subplot(231); imagesc(axi_proj);   colormap gray; axis off image;
    subplot(232); imagesc(cor_proj);   colormap gray; axis off image;title('gp');
    subplot(233); imagesc(sag_proj);   colormap gray; axis off image;
    subplot(234); imagesc(axi_recons); colormap gray; axis off image;
    subplot(235); imagesc(cor_recons); colormap gray; axis off image;title('recon');
    subplot(236); imagesc(sag_recons); colormap gray; axis off image;
    pause();
end