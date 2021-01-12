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

stype     = '20181120_NIH3T3_LipidDroplet(PM)_0.313.30/';
sname     = '20181120.190011.239.Default-092/';

root = ['../../GP_recon/bio/' stype sname];
root_save = '../../dataset/bio/';
root_save_input = [root_save 'Input_infer_axi/' stype sname]; mkdir(root_save_input);
root_save_input_sag = [root_save 'Input_infer_sag/' stype sname]; mkdir(root_save_input_sag);
root_save_input_cor = [root_save 'Input_infer_cor/' stype sname]; mkdir(root_save_input_cor);

% Load data
load([root 'RI_NN'], 'RI_tomogram');
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
    save([root_save_input 'a' num2str(i)], 'proj');
end

disp('saving saggital projection');
for i = 1:360
    proj = gp_proj2(:, :, i);
    save([root_save_input_sag 'a' num2str(i)], 'proj');
end

disp('saving coronal projection');
for i = 1:360
    proj = gp_proj3(:, :, i);
    save([root_save_input_cor 'a' num2str(i)], 'proj');
end