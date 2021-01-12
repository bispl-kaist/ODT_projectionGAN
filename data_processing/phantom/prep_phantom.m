clear; clc;
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

I_save_dir      = '../data/TomoGAN_db_infer_multiball/Input/';

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

for i = 8:length(gp_dir)
    disp(i);
    load([gp_root gp_dir(i).name], 'RI_tomogram');
    I_save_diri = [I_save_dir sprintf('%03d',i) '/'];
    mkdir(I_save_diri);
    load([gp_root gp_dir(i).name], 'RI_tomogram');
    gp_tomo = abs(RI_tomogram); Gmax = max(gp_tomo(:)); Gmin = min(gp_tomo(:));
    
    gp_tomo             = normalize_im(gp_tomo);
    gp_proj             = A(permute(gp_tomo, [2 3 1]));
    
    for a = 1:360
        input = squeeze(gp_proj(:,:,a));
        save([I_save_diri 'p' int2str(i) '_ang' int2str(a)], 'input');
    end
end