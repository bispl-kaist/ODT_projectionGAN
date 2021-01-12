clear; clc;

restoredefaultpath();
addpath('./utils');
addpath('./tomo_func');

%% System parameters
% %% Parameter Setting
DSO             = 400;                      % [mm]
DSD             = 800;                     % [mm]

% %% Make Object
pdImgSize       = [372, 372, 372];          % [mm x mm]
pnImgSize       = [372, 372, 372];

% %% Make Detector
pdStepDct     	= 1;                        % [mm]
pnSizeDct       = [528, 528];               % [elements]
% pnSizeDct       = [372, 372];               % [elements]

pdOffsetDct     = 0;                        % [elements]

% %% Rotation Setup
nNumView        = 360;                      % [elements]
dStepView       = 2*pi/360;                 % [radian]

% %% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);


%% Definition of Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/(2*nNumView));
AINV            = @(y) BackProjection(Filtering(y, param), param);
ATA             = @(x) AT(A(x));

%% Load data, and apply projection

root = '../../GP_recon/microbead/';
root_save = '../../dataset/microbead2/';
root_save_input_axi = [root_save 'Input_microbead_axi/'];
root_save_input_sag = [root_save 'Input_microbead_sag/'];
root_save_input_cor = [root_save 'Input_micobead_cor/'];

outer_dir = dir(root);

specimen_fname = '20180726.150404.579.SiO2_5um-001';
disp(specimen_fname);

root_s = [root specimen_fname '/'];
root_save_input_axi_s = [root_save_input_axi specimen_fname '/'];
root_save_input_sag_s = [root_save_input_sag specimen_fname '/'];
root_save_input_cor_s = [root_save_input_cor specimen_fname '/'];
mkdir(root_save_input_axi_s);
mkdir(root_save_input_sag_s);
mkdir(root_save_input_cor_s);

load([root_s 'RI_NN'], 'RI_tomogram');
gp_tomo = abs(RI_tomogram);
[norm_gp_tomo, maxv, minv] = normalize_im_verbose(gp_tomo);

% projection [1 2 3]
gp_proj = A(norm_gp_tomo);

% projection [3 1 2]
norm_gp_tomo2 = permute(norm_gp_tomo, [3 1 2]);
gp_proj2 = A(norm_gp_tomo2);

% projection [2 3 1]
norm_gp_tomo3 = permute(norm_gp_tomo, [2 3 1]);
gp_proj3 = A(norm_gp_tomo3);

disp('saving axial projection');
for i = 1:360
    proj = gp_proj(:, :, i);
    save([root_save_input_axi_s 'a' num2str(i)], 'proj');
end

disp('saving saggital projection');
for i = 1:360
    proj = gp_proj2(:, :, i);
    save([root_save_input_sag_s 'a' num2str(i)], 'proj');
end

disp('saving coronal projection');
for i = 1:360
    proj = gp_proj3(:, :, i);
    save([root_save_input_cor_s 'a' num2str(i)], 'proj');
end