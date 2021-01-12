function mra_out = manipulate_MRA(mra_in, max_th, min_th)

% Clip values

mra_in(mra_in > max_th) = max_th;
mra_in(mra_in < min_th) = min_th;

% Log scale
log_mra_in = log(mra_in);
log_max = max(log_mra_in(:));
log_min = min(log_mra_in(:));

orig_range = max_th - min_th;
log_range = log_max - log_min;
ratio = orig_range / log_range;

rescaled_log_mra_in = (log_mra_in - log_min) * ratio + min_th;

mra_out = (3 * mra_in + rescaled_log_mra_in) / 4;

end