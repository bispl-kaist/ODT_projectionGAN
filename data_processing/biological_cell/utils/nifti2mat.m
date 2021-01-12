function nifti2mat(fname)
% Loads a .nii file format, and saves it to the same directory with a
% .mat format
% Use after applying BET with FSL, before MIP in multiple directions

niftiread(fname);
save(fname);

end