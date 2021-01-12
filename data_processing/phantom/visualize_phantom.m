clear; clc;
root = '../data/ODT_phantoms/';
addpath('utils');
addpath('lib_util');
addpath('lib_optim');
addpath('tomo_func');

label_root      = [root 'label/'];
label_dir       = dir(label_root);
label_dir       = label_dir(3:end);
gp_root         = [root 'gp/'];
gp_dir          = dir(gp_root);
gp_dir          = gp_dir(3:end);

% TomoGAN_Input_dir        = '../data/TomoGAN_db_infer/Input/';
TomoGAN_Input_dir        = '../data/TomoGAN_db_infer_multiball/Input/';
TomoGAN_Recon_dir        = '../results_patch/';

method = 'patch192_g1e-2_phantom_MB_multiD_ident_all_stride24/';
TomoGAN_Recon_dir        = [TomoGAN_Recon_dir method 'recon_axi/'];

% sidx                     = 1:27; % Which phantom to visualize
sidx                     = 1;

%%
figure(18); 
for s = sidx
    disp(s);
    for i = 1:5:360
        fname = ['p' int2str(s) '_ang' int2str(i)];
%         load([TomoGAN_Input_dir int2str(s) '/' fname], 'input');
%         load([TomoGAN_Recon_dir int2str(s) '/' fname], 'recons');
        load([TomoGAN_Input_dir sprintf('%03d',s) '/' fname], 'input');
        load([TomoGAN_Recon_dir sprintf('%03d',s) '/' fname], 'recons');
        recons = squeeze(recons);
        
        sgtitle(i);
        subplot(121); imagesc(input); colormap gray; axis off image;
        subplot(122); imagesc(recons); colormap gray; axis off image;
        pause();
    end
end

