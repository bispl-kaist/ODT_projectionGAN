clear; clc;
% root = '../data/ODT_phantoms/';
root = '../data/ODT_phantoms_multiball/';
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

TomoGAN_Recon_dir        = '../results/';
method = 'patch192_g1e-2_phantom_MB_multiD_ident/';
TomoGAN_Recon_dir        = [TomoGAN_Recon_dir method 'recon_axi/'];

save_dir_vtk = ['../results_vtk/' method];
mkdir(save_dir_vtk);

save_dir        = '/home/harry/Documents/2020-1/research/ODT/0930/';
mkdir(save_dir);
save_dir_m      = [save_dir method];
mkdir(save_dir_m);



%% System parameters
% %% Parameter Setting
DSO             = 400;                      % [mm]
DSD             = 1000;                     % [mm]

% %% Make Object
pdImgSize       = [372, 372, 372];          % [mm x mm]
pnImgSize       = [372, 372, 372];

% %% Make Detector
pdStepDct     	= 1;                        % [mm]
pnSizeDct       = [512, 512];               % [elements]

pdOffsetDct     = 0;                        % [elements]

% %% Rotation Setup
nNumView        = 360;                       % [elements]
dStepView       = 2*pi/360;              % [radian]

% %% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);

        
%% Definition of Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));
AINV            = @(y) BackProjection(Filtering(y, param), param);
ATA             = @(x) AT(A(x));


%%
sidx                     = 1; % Which phantom to visualize
for i = sidx
    disp(i);
    recons_proj = zeros(512, 512, 360);
    
    if contains(label_dir(i).name, '001') || contains(label_dir(i).name, '002') 
        load([label_root label_dir(i).name], 'ph');
        label_tomo = abs(ph); Lmax = max(label_tomo(:)); Lmin = min(label_tomo(:));
    else
        load([label_root label_dir(i).name], 'phantom_cube');
        label_tomo = phantom_cube; Lmax = max(label_tomo(:)); Lmin = min(label_tomo(:));
    end
    load([gp_root gp_dir(i).name], 'RI_tomogram');
    gp_tomo = abs(RI_tomogram); Gmax = max(gp_tomo(:)); Gmin = min(gp_tomo(:));
    
    gp_tomo             = normalize_im(gp_tomo);
    gp_proj             = A(permute(gp_tomo, [2 3 1]));
    for a = 1:360
        fname = ['p' int2str(i) '_ang' int2str(a)];
        load([TomoGAN_Recon_dir int2str(i) '/' fname], 'recons');
        recons(isnan(recons)) = 0;
        recons_proj(:, :, a) = squeeze(recons);
        recons_proj(recons_proj < 0) = 0;
    end
    recons = AINV(recons_proj);
    max_recons = max(recons(:));
    recons(recons < max_recons * 0.1) = 0;
    recons = denormalize_im(permute(recons, [3 1 2]), Gmax, Gmin);
    
    save_dir_m_i       = [save_dir_m int2str(i)]; mkdir(save_dir_m_i);
    save([save_dir_m_i '/recons'], 'recons');
end