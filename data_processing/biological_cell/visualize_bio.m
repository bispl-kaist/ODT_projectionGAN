%% Load data, and apply projection

root_save = '/media/harry/fastmri/ODT_conventional_tomo_inference_db/';
root_input_axi = [root_save 'Input_infer_axi/'];
root_input_cor = [root_save 'Input_infer_cor/'];
root_input_sag = [root_save 'Input_infer_sag/'];
root_recon = '/media/harry/ExtDrive/ODT_conventional_tomo/results/';
root_recon_vol = '/media/harry/ExtDrive/ODT_conventional_tomo/results_vol/';
method = 'tomo_patch_g5/';
root_recon = [root_recon method];
root_recon_axi = [root_recon 'recon_axi/'];
root_recon_cor = [root_recon 'recon_cor/'];
root_recon_sag = [root_recon 'recon_sag/'];

specimen_type = '20181120_NIH3T3_LipidDroplet(PM)_0.313.30/';
specimen_name = '20181120.150103.398.Default-062/';

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
    subplot(231); imagesc(axi_proj);   colormap gray; axis off image;title(i);
    subplot(232); imagesc(cor_proj);   colormap gray; axis off image;title(i);
    subplot(233); imagesc(sag_proj);   colormap gray; axis off image;title(i);
    subplot(234); imagesc(axi_recons); colormap gray; axis off image;
    subplot(235); imagesc(cor_recons); colormap gray; axis off image;
    subplot(236); imagesc(sag_recons); colormap gray; axis off image;
            pause();
%     drawnow();
end