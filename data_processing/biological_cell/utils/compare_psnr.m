function psnr_val = compare_psnr(label, recon)
    label = label / max(label(:));
    recon = recon / max(recon(:));
    psnr_val = psnr(recon, label, max(label(:)));
end