clear; clc;

restoredefaultpath();
addpath('./utils');
addpath('./tomo_func');

%% System parameters
% %% Parameter Setting
DSO             = 400;                      % [mm]
DSD             = 1000;                     % [mm]

% %% Make Object
pdImgSize       = [372, 372, 372];          % [mm x mm]
pnImgSize       = [372, 372, 372];

% %% Make Detector
pdStepDct     	= 1;                        % [mm]
pnSizeDct       = [528, 528];               % [elements]

pdOffsetDct     = 0;                        % [elements]

% %% Rotation Setup
nNumView        = 360;                      % [elements]
dStepView       = 2*pi/360;                 % [radian]

% %% Make Object (Image, Detector)
param           = MakeParam(pdImgSize, pnImgSize, pdStepDct, pnSizeDct, pdOffsetDct, dStepView, nNumView, DSO, DSD);


%% Definition of Operators
A               = @(x) Projection(x, param);
AT              = @(y) BackProjection(y, param)/(pi/2*(nNumView));
AINV            = @(y) BackProjection(Filtering(y, param), param);
ATA             = @(x) AT(A(x));

%% Load data, and apply projection

root_input = '../../GP_recon/microbead/';
root_recon = '../../recons/microbead/';
root_recon_axi = [root_recon 'recon_microbead_axi/Input_microbead_axi/'];
root_recon_cor = [root_recon 'recon_microbead_cor/Input_microbead_axi/'];
root_recon_sag = [root_recon 'recon_microbead_sag/Input_microbead_axi/'];
root_save = '../../recons_vol/microbead/';
mkdir(root_save);

sname = '20180726.150404.579.SiO2_5um-001/';
root_input = [root_input sname];

% Load Input data
disp('loading input data');
load([root_input 'RI_NN'], 'RI_tomogram');
gp_tomo = RI_tomogram;
gp_tomo = permute(abs(gp_tomo), [3 1 2]);
[norm_gp_tomo, maxv, minv] = normalize_im_verbose(gp_tomo);

recons_tomo = zeros(372, 372, 372);
% Axial Recon data FBP
disp('loading axial recon data');
recons_proj = zeros(528, 528, 360);
for i = 1:360
    load([root_recon_axi 'a' num2str(i)], 'recons');
    recons(recons<0) = 0;
    recons_proj(:,:,i) = recons;
end

% FBP
axi_recons_tomo = AINV(recons_proj);
axi_recons_tomo(axi_recons_tomo < 0) = 0;
axi_recons_tomo = permute(axi_recons_tomo, [3 1 2]);
axi_recons_tomo = denormalize_im(axi_recons_tomo, maxv, minv);
recons_tomo = recons_tomo + axi_recons_tomo;

% Coronal Recon data FBP
disp('loading coronal recon data');
for i = 1:360
    load([root_recon_cor 'a' num2str(i)], 'recons');
    recons(recons<0) = 0;
    recons_proj(:,:,i) = recons;
end

% FBP
cor_recons_tomo = AINV(recons_proj);
cor_recons_tomo(cor_recons_tomo < 0) = 0;
cor_recons_tomo = permute(cor_recons_tomo, [2 3 1]);
cor_recons_tomo = denormalize_im(cor_recons_tomo, maxv, minv);
recons_tomo = recons_tomo + cor_recons_tomo;

% Sagittal Recon data FBP
disp('loading sagittal recon data');
for i = 1:360
    load([root_recon_cor 'a' num2str(i)], 'recons');
    recons(recons<0) = 0;
    recons_proj(:,:,i) = recons;
end

% FBP
sag_recons_tomo = AINV(recons_proj);
sag_recons_tomo(sag_recons_tomo < 0) = 0;
sag_recons_tomo = permute(sag_recons_tomo, [2 3 1]);
sag_recons_tomo = denormalize_im(sag_recons_tomo, maxv, minv);
recons_tomo = recons_tomo + sag_recons_tomo;
recons_tomo = recons_tomo ./ 3;

%% Save slice data as img
disp('Saving');
save([root_save 'recons_tomo'], 'recons_tomo');