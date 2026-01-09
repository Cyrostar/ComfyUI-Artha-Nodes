import cv2
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# ----------------------------
# Utility color space helpers
# ----------------------------

def rgb_to_lab(img):
    return cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

def lab_to_rgb(lab_img):
    lab_uint8 = np.clip(lab_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

def rgb_to_yuv(img):
    return cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0

def yuv_to_rgb(yuv):
    return cv2.cvtColor((np.clip(yuv, 0, 1) * 255).astype(np.uint8), cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
    
def rgb_to_xyz(rgb):
    def gamma_correct(c):
        return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))
    linear_rgb = gamma_correct(rgb)
    xyz = np.zeros_like(rgb)
    xyz[:,:,0] = 0.4124564 * linear_rgb[:,:,0] + 0.3575761 * linear_rgb[:,:,1] + 0.1804375 * linear_rgb[:,:,2]
    xyz[:,:,1] = 0.2126729 * linear_rgb[:,:,0] + 0.7151522 * linear_rgb[:,:,1] + 0.0721750 * linear_rgb[:,:,2]
    xyz[:,:,2] = 0.0193339 * linear_rgb[:,:,0] + 0.1191920 * linear_rgb[:,:,1] + 0.9503041 * linear_rgb[:,:,2]
    return xyz
    
def xyz_to_rgb(xyz):
    linear_rgb = np.zeros_like(xyz)
    linear_rgb[:,:,0] = 3.2404542 * xyz[:,:,0] - 1.5371385 * xyz[:,:,1] - 0.4985314 * xyz[:,:,2]
    linear_rgb[:,:,1] = -0.9692660 * xyz[:,:,0] + 1.8760108 * xyz[:,:,1] + 0.0415560 * xyz[:,:,2]
    linear_rgb[:,:,2] = 0.0556434 * xyz[:,:,0] - 0.2040259 * xyz[:,:,1] + 1.0572252 * xyz[:,:,2]
    def inverse_gamma_correct(c):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1/2.4) - 0.055)
    rgb = inverse_gamma_correct(linear_rgb)
    return np.clip(rgb, 0, 1)

def xyz_to_lab(xyz):
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    fx = xyz[:,:,0] / xn
    fy = xyz[:,:,1] / yn
    fz = xyz[:,:,2] / zn
    def f_transform(t):
        return np.where(t > (6/29)**3, np.power(t, 1/3), (1/3) * ((29/6)**2) * t + (4/29))
    fx = f_transform(fx)
    fy = f_transform(fy)
    fz = f_transform(fz)
    lab = np.zeros_like(xyz)
    lab[:,:,0] = 116 * fy - 16
    lab[:,:,1] = 500 * (fx - fy)
    lab[:,:,2] = 200 * (fy - fz)
    return lab
    
def lab_to_xyz(lab):
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    def inv_f_transform(t):
        return np.where(t > 6/29, t**3, 3 * ((6/29)**2) * (t - 4/29))
    X = xn * inv_f_transform(fx)
    Y = yn * inv_f_transform(fy)
    Z = zn * inv_f_transform(fz)
    xyz = np.stack([X, Y, Z], axis=-1)
    return xyz
    
def detect_gamut(rgb):
    flat_rgb = rgb.reshape(-1, 3)
    max_val = np.max(flat_rgb)
    mean_sat = np.mean(np.max(flat_rgb, axis=1) - np.min(flat_rgb, axis=1))
    
    if max_val > 1.0 or mean_sat > 0.7:
        return 'rec2020'
    elif mean_sat > 0.5:
        return 'adobe_rgb'
    else:
        return 'srgb'
          
# ----------------------------
# Transfer methods
# ----------------------------

def reinhard_color_transfer(source, target):
    """Classic mean/std transfer in OpenCV LAB (0..255 units)."""
    # Expect source/target in [0,1] float RGB
    src_lab = cv2.cvtColor((np.clip(source,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor((np.clip(target,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

    src_mean, src_std = src_mean.flatten(), src_std.flatten()
    tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

    result_lab = (src_lab - src_mean) * (tgt_std / (src_std + 1e-8)) + tgt_mean
    return lab_to_rgb(result_lab)


def mkl_color_transfer(source, target):
    """MKL-like transform on LAB (uses rgb_to_lab which returns 0..255 LAB)."""
    try:
        s_lab = rgb_to_lab(source).reshape(-1, 3)
        t_lab = rgb_to_lab(target).reshape(-1, 3)

        cov_src = np.cov(s_lab, rowvar=False)
        cov_tgt = np.cov(t_lab, rowvar=False)

        # Regularize for numerical stability
        cov_src += 1e-5 * np.eye(3)
        cov_tgt += 1e-5 * np.eye(3)

        Cs_half = np.linalg.cholesky(cov_src)
        Ct_half = np.linalg.cholesky(cov_tgt)

        A = Ct_half @ np.linalg.inv(Cs_half)

        mean_src, mean_tgt = np.mean(s_lab, axis=0), np.mean(t_lab, axis=0)
        transferred = (s_lab - mean_src) @ A.T + mean_tgt

        return lab_to_rgb(transferred.reshape(source.shape))
    except Exception:
        return reinhard_color_transfer(source, target)
        
def ciede2000_color_transfer(source, target):
    """CIEDE2000-based perceptually uniform color transfer"""
    source_xyz = rgb_to_xyz(source)
    target_xyz = rgb_to_xyz(target)
    source_lab = xyz_to_lab(source_xyz)
    target_lab = xyz_to_lab(target_xyz)
    source_lab_flat = source_lab.reshape(-1, 3)
    target_lab_flat = target_lab.reshape(-1, 3)
    n_samples = min(1000, len(source_lab_flat), len(target_lab_flat))
    target_sample = target_lab_flat[np.random.choice(len(target_lab_flat), n_samples, replace=False)]
    result_lab_flat = np.zeros_like(source_lab_flat)
    chunk_size = 100
    for i in range(0, len(source_lab_flat), chunk_size):
        end_i = min(i + chunk_size, len(source_lab_flat))
        chunk = source_lab_flat[i:end_i]
        chunk_result = np.zeros_like(chunk)
        for j, pixel in enumerate(chunk):
            distances = np.sum((target_sample - pixel)**2, axis=1)
            best_match_idx = np.argmin(distances)
            chunk_result[j] = target_sample[best_match_idx]
        result_lab_flat[i:end_i] = chunk_result
    result_lab = result_lab_flat.reshape(source.shape)
    result_xyz = lab_to_xyz(result_lab)
    result_rgb = xyz_to_rgb(result_xyz)
    return np.clip(result_rgb, 0, 1)

def lab_nn_transfer(source, target, sample_size=600, chunk=2048, k=3, smoothing=True):

    # Convert to LAB in 0..255
    src_lab = cv2.cvtColor((source * 255).astype("uint8"), cv2.COLOR_RGB2LAB).astype("float32")
    tgt_lab = cv2.cvtColor((target * 255).astype("uint8"), cv2.COLOR_RGB2LAB).astype("float32")

    # Flatten channels
    s_L, s_A, s_B = cv2.split(src_lab)
    t_L, t_A, t_B = cv2.split(tgt_lab)

    s_ab = np.stack([s_A.flatten(), s_B.flatten()], axis=1)
    t_ab = np.stack([t_A.flatten(), t_B.flatten()], axis=1)

    # Sample target AB
    M = min(sample_size, len(t_ab))
    idx = np.random.choice(len(t_ab), size=M, replace=False)
    t_sample = t_ab[idx]

    # Fit NN on target AB
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(t_sample)

    mapped_ab = []
    for i in range(0, len(s_ab), chunk):
        dists, ids = nn.kneighbors(s_ab[i:i+chunk])
        weights = 1.0 / (dists + 1e-6)  # inverse distance weights
        weights /= weights.sum(axis=1, keepdims=True)
        weighted = np.sum(t_sample[ids] * weights[..., None], axis=1)
        mapped_ab.append(weighted)

    mapped_ab = np.vstack(mapped_ab).reshape(s_A.shape + (2,))

    # Recombine with original luminance
    mapped_lab = np.stack([s_L, mapped_ab[..., 0], mapped_ab[..., 1]], axis=-1)

    # Convert back to RGB
    result = cv2.cvtColor(mapped_lab.astype("uint8"), cv2.COLOR_LAB2RGB)
    result = np.clip(result.astype("float32") / 255.0, 0, 1)

    # Optional bilateral smoothing to reduce artifacts
    if smoothing:
        result = cv2.bilateralFilter((result * 255).astype("uint8"), d=7, sigmaColor=50, sigmaSpace=50)
        result = result.astype("float32") / 255.0

    return result

def color_mood_transfer(source, target):
    """Simple color scaling based on per-channel means (fast, artistic)."""
    s_mean = np.mean(source, axis=(0, 1))
    t_mean = np.mean(target, axis=(0, 1))
    scale = t_mean / (s_mean + 1e-8)
    return np.clip(source * scale, 0, 1)


def pdf_transfer_method(source, target, bins=64):
    """Per-channel histogram CDF matching. Inputs in [0,1]."""
    matched = np.zeros_like(source)
    for ch in range(3):
        src_hist, edges = np.histogram(source[:, :, ch], bins=bins, range=(0, 1), density=True)
        tgt_hist, _ = np.histogram(target[:, :, ch], bins=bins, range=(0, 1), density=True)

        src_cdf = np.cumsum(src_hist)
        src_cdf = src_cdf / (src_cdf[-1] + 1e-12)
        tgt_cdf = np.cumsum(tgt_hist)
        tgt_cdf = tgt_cdf / (tgt_cdf[-1] + 1e-12)

        # Map source values through its CDF then invert target CDF
        src_vals = np.interp(source[:, :, ch].flatten(), edges[:-1], src_cdf)
        mapped = np.interp(src_vals, tgt_cdf, edges[:-1])
        matched[:, :, ch] = mapped.reshape(source.shape[:2])
    return np.clip(matched, 0, 1)

def regrain(result_rgb, source_rgb, strength=0.6, radius=3):
    """Restore high-frequency luminance detail from source into result."""
    src_yuv = rgb_to_yuv(source_rgb)
    res_yuv = rgb_to_yuv(result_rgb)
    src_lum = src_yuv[:, :, 0]
    blur = cv2.GaussianBlur(src_lum, (radius|1, radius|1), 0)  # ensure odd kernel
    detail = src_lum - blur
    res_yuv[:, :, 0] = np.clip(res_yuv[:, :, 0] + strength * detail, 0, 1)
    return yuv_to_rgb(res_yuv)

# ----------------------------
# Mixing strategies
# ----------------------------

def luminance_split_mixing(source, target, method_fn):
    result = method_fn(source, target)
    src_yuv, res_yuv = rgb_to_yuv(source), rgb_to_yuv(result)
    res_yuv[:, :, 0] = src_yuv[:, :, 0]
    return yuv_to_rgb(res_yuv)

def detail_preserve_mixing(source, target, method_fn):
    result = method_fn(source, target)
    src_yuv, res_yuv = rgb_to_yuv(source), rgb_to_yuv(result)
    detail = src_yuv[:, :, 0] - cv2.GaussianBlur(src_yuv[:, :, 0], (3, 3), 0)
    res_yuv[:, :, 0] = np.clip(res_yuv[:, :, 0] + detail, 0, 1)
    return yuv_to_rgb(res_yuv)

# ----------------------------
# Dispatcher
# ----------------------------

def apply_single_method(source, target, method, gamut_mode):
    
    if isinstance(source, torch.Tensor):
        source = source[0].cpu().numpy().astype("float32")
    if isinstance(target, torch.Tensor):
        target = target[0].cpu().numpy().astype("float32")
        
    if method == "reinhard":
        return reinhard_color_transfer(source, target)
    elif method == "mkl":
        return mkl_color_transfer(source, target)
    elif method == "ciede2000":
        return ciede2000_color_transfer(source, target)
    elif method == "lab_nn":
        return lab_nn_transfer(source, target)
    elif method == "color_mood":
        return color_mood_transfer(source, target)
    elif method == "pdf":
        return pdf_transfer_method(source, target)
    else:
        raise ValueError(f"Unknown method {method}")