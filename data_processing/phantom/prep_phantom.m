clear; clc;

restoredefaultpath();
addpath('./utils');
addpath('./tomo_func');


%%

clear; clc;
root = '../../GP_recon/phantom/';
snum   = 1;
I_save_dir      = ['../../dataset/phantom/' sprintf('%03d',snum) '/']; mkdir(I_save_dir);

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

load([root sprintf('phantom_%03d',snum)], 'RI_tomogram');
gp_tomo = abs(RI_tomogram); Gmax = max(gp_tomo(:)); Gmin = min(gp_tomo(:));
gp_tomo             = normalize_im(gp_tomo);
gp_proj             = A(permute(gp_tomo, [2 3 1]));

for a = 1:360
    input = squeeze(gp_proj(:,:,a));
    save([I_save_dir 'p1_ang' int2str(a)], 'input');
end