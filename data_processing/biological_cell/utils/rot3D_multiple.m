function rot_vol_stack = rot3D_multiple(vol, fname, axis)

% Rotates 3D data with respect to a given axis - 'roll' or 'spin'
% 13 views - 0 : 15 : 180 degrees
if strcmp(axis, 'roll')
    
    save_fname = [fname '_roll'];
    
    axis = [1 0 0];
%     rot_vol_stack = zeros(561,528,13);
    rot_vol_stack = zeros(587,528,13);
    for i = 1:13
        deg = - 15 * i + 5;
        rot_vol = imrotate3(vol, deg, axis);
        [h, w, c] = size(rot_vol);
        if i >= 5 && i <= 9
            rot_vol = imresize(rot_vol, [round(h * 1.2) w]);
            [h, w, c] = size(rot_vol);
        end
        margin = 587 - h;
        rot_vol_MIP = squeeze(max(permute(rot_vol, [3 1 2])));
        if mod(margin,2) == 0
            rot_vol_stack(ceil(margin / 2)+1:end - ceil(margin / 2),:,i) = rot_vol_MIP;
        elseif mod(margin,2) == 1
            rot_vol_stack(ceil(margin / 2):end - ceil(margin / 2),:,i) = rot_vol_MIP;
        end
    end
elseif strcmp(axis, 'spin')
    
    save_fname = [fname '_spin'];
    
    vol = permute(vol, [3 2 1]);
    axis = [0 1 0];
    rot_vol_stack = zeros(195,747,13);
    for i = 1:13
%         deg = -15 * i + 15;
        deg = 15 * (i+1) - 15;
        rot_vol = imrotate3(vol, deg, axis);
        rot_vol = flipud(rot_vol);
        
        [h, w, c] = size(rot_vol);
        margin = 747 - w;
        rot_vol_MIP = squeeze(max(permute(rot_vol, [3 1 2])));
        if mod(margin,2) == 0
            rot_vol_stack(:,ceil(margin / 2)+1:end - ceil(margin / 2),i) = rot_vol_MIP;
        elseif mod(margin,2) == 1
            rot_vol_stack(:,ceil(margin / 2):end - ceil(margin / 2),i) = rot_vol_MIP;
        end
    end
    rot_vol_stack = imresize(rot_vol_stack, [250 747], 'bilinear');
end

for i = 1:13
    MIP = rot_vol_stack(:,:,i);
    save_img(MIP, [save_fname int2str(i) '.png']);
end

rot_vol_stack = permute(rot_vol_stack, [1 2 4 3]);
dicomwrite(uint16(rot_vol_stack), save_fname);
rot_vol_stack = squeeze(rot_vol_stack);

end