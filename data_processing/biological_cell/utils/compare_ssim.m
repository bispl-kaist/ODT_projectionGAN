function ssim_val = compare_ssim(label, recon)
    label = label / max(label(:));
    recon = recon / max(recon(:));
    ssim_val = ssim(recon, label, 'DynamicRange', max(label(:)));
end