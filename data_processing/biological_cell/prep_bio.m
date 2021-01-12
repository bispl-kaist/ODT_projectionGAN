clear; clc;

addpath('./lib_util');
addpath('./lib_optim');
addpath('./utils');

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

root = './ODT_DATA_gp/';
root_save = '/media/harry/fastmri/ODT_conventional_tomo_inference_db/';
root_save_input = [root_save 'Input_infer_axi/'];
root_save_input_sag = [root_save 'Input_infer_sag/'];
root_save_input_cor = [root_save 'Input_infer_cor/'];
root_save_label = [root_save 'Label/'];

outer_dir = dir(root);

% for s = 3:length(outer_dir)
% s_list = [4, 6, 7];
s_list = [4];
for s = s_list
    disp(outer_dir(s).name);
    root_s = [root outer_dir(s).name '/'];
    root_save_input_s = [root_save_input outer_dir(s).name '/'];
    root_save_input_sag_s = [root_save_input_sag outer_dir(s).name '/'];
    root_save_input_cor_s = [root_save_input_cor outer_dir(s).name '/'];
    root_save_label_s = [root_save_label outer_dir(s).name '/'];
    mkdir(root_save_input_s);
    mkdir(root_save_input_sag_s);
    mkdir(root_save_input_cor_s);
    mkdir(root_save_label_s);
    
    inner_dir = dir(root_s);
    for ss = 42      
        disp(inner_dir(ss).name);
        root_ss = [root_s inner_dir(ss).name '/'];
        root_save_input_ss = [root_save_input_s inner_dir(ss).name '/'];
        root_save_input_sag_ss = [root_save_input_sag_s inner_dir(ss).name '/'];
        root_save_input_cor_ss = [root_save_input_cor_s inner_dir(ss).name '/'];
        root_save_label_ss = [root_save_label_s inner_dir(ss).name '/'];
        mkdir(root_save_input_ss);
        mkdir(root_save_input_sag_ss);
        mkdir(root_save_input_cor_ss);
        mkdir(root_save_label_ss);
        % Load data
        load([root_ss 'RI_NN'], 'RI_tomogram');
        gp_tomo = abs(RI_tomogram);
        [norm_gp_tomo, maxv, minv] = normalize_im(gp_tomo);
        
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
            save([root_save_input_ss 'a' num2str(i)], 'proj');
        end
        
        disp('saving saggital projection');
        for i = 1:360
            proj = gp_proj2(:, :, i);
            save([root_save_input_sag_ss 'a' num2str(i)], 'proj');
        end
        
        disp('saving coronal projection');
        for i = 1:360
            proj = gp_proj3(:, :, i);
            save([root_save_input_cor_ss 'a' num2str(i)], 'proj');
        end
    end
end