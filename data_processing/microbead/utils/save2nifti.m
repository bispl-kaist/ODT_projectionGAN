function save2nifti(fname)
% Loads a .mat file format, and saves it to the same directory with a
% .nii format
% Use before applying BET with FSL
% The variable name must be 'MRA_stack'
% Otherwise throw an error

load(fname, 'MRA_stack');
niftiwrite(MRA_stack, fname);

end