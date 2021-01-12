function zeroed_c = zero_out_wavelet_component(img, level, direction)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% direction: 'vertical' or 'horizontal'
% Assumes homogeneous shape
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[c, s] = wavedec2(img, level, wav);

for i = level:-1:1
    len = s(end-i, 1); % length of the corresponding level img
    
