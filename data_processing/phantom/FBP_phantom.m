clear; clc;

restoredefaultpath();
addpath('./utils');
addpath('./tomo_func');

snum       = 1;

%%
gp_root         = ['../../GP_recon/phantom/'];
recon_root      = ['../../recons/phantom/recon_axi/' sprintf('%03d/', snum)];
save_root       = ['../../recons_vol/phantom/' sprintf('%03d/',snum)]; mkdir(save_root);


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
recons_proj = zeros(512, 512, 360);
load([gp_root sprintf('phantom_%03d',snum)], 'RI_tomogram');
gp_tomo = abs(RI_tomogram); Gmax = max(gp_tomo(:)); Gmin = min(gp_tomo(:));

gp_tomo             = normalize_im(gp_tomo);
gp_proj             = A(permute(gp_tomo, [2 3 1]));
for a = 1:360
    fname = ['p' int2str(snum) '_ang' int2str(a)];
    load([recon_root fname], 'recons');
    recons(isnan(recons)) = 0;
    recons_proj(:, :, a) = squeeze(recons);
    recons_proj(recons_proj < 0) = 0;
end
recons = AINV(recons_proj);
max_recons = max(recons(:));
recons(recons < max_recons * 0.1) = 0;
recons = denormalize_im(permute(recons, [3 1 2]), Gmax, Gmin);

save([save_root sprintf('phantom_%03d',snum)],'recons');