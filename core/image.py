import os
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import io
import base64

def load_pil_image(image_path, auto_orient=True):
    """Load PIL image from path with optional auto-orientation"""
    img = Image.open(image_path)
    
    # Auto-orient based on EXIF data
    if auto_orient:
        img = ImageOps.exif_transpose(img)
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def resize_image_proportional(img, target_width, target_height, keep_proportions=True, upscale_method="lanczos"):
    """Resize image with optional proportion preservation"""
    if not keep_proportions:
        # Stretch to exact dimensions (may distort image)
        return img.resize((target_width, target_height), 
                         getattr(Image.Resampling, upscale_method.upper()))
    
    # Calculate new size maintaining aspect ratio
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height
    
    # Determine which dimension to use as reference
    if (target_width / target_height) > aspect_ratio:
        # Height is the limiting factor
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        # Width is the limiting factor
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    
    return img.resize((new_width, new_height), getattr(Image.Resampling, upscale_method.upper()))


def crop_image_to_aspect(img, target_width, target_height, upscale_method="lanczos"):
    """Crop image to target aspect ratio then resize"""
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height
    target_aspect = target_width / target_height
    
    if aspect_ratio > target_aspect:
        # Image is wider, crop width
        new_width = int(original_height * target_aspect)
        left = (original_width - new_width) // 2
        img = img.crop((left, 0, left + new_width, original_height))
    else:
        # Image is taller, crop height
        new_height = int(original_width / target_aspect)
        top = (original_height - new_height) // 2
        img = img.crop((0, top, original_width, top + new_height))
    
    return img.resize((target_width, target_height), 
                     getattr(Image.Resampling, upscale_method.upper()))


def pad_image_to_dimensions(img, target_width, target_height, upscale_method="lanczos", pad_color=(0, 0, 0)):
    """Pad image to target dimensions while maintaining aspect ratio"""
    # First resize to fit within target dimensions
    img.thumbnail((target_width, target_height), 
                 getattr(Image.Resampling, upscale_method.upper()))
    
    # Create new image with target dimensions and paste centered
    padded_img = Image.new('RGB', (target_width, target_height), pad_color)
    paste_x = (target_width - img.width) // 2
    paste_y = (target_height - img.height) // 2
    padded_img.paste(img, (paste_x, paste_y))
    
    return padded_img
    
def rotate_image(img, rotation_angle, expand=True):
    """Rotate image by specified angle"""
    if rotation_angle == 0:
        return img
    elif rotation_angle == 90:
        return img.rotate(90, expand=expand)
    elif rotation_angle == 180:
        return img.rotate(180, expand=expand)
    elif rotation_angle == 270:
        return img.rotate(270, expand=expand)
    else:
        # For arbitrary angles, use high-quality rotation
        return img.rotate(rotation_angle, resample=Image.Resampling.BICUBIC, expand=expand, fillcolor=(0, 0, 0))

def flip_mirror_image(img, flip_horizontal=False, flip_vertical=False):
    """Apply horizontal and/or vertical flips to image"""
    result_img = img
    
    if flip_horizontal:
        result_img = result_img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if flip_vertical:
        result_img = result_img.transpose(Image.FLIP_TOP_BOTTOM)
    
    return result_img

def pil_to_tensor(img, normalize=True):
    """Convert PIL Image to ComfyUI tensor format"""
    # Convert PIL Image to numpy array
    image_array = np.array(img).astype(np.float32)
    
    # Normalize to [0,1] range if requested
    if normalize:
        image_array /= 255.0
    
    # Convert to torch tensor and adjust dimensions for ComfyUI (B, H, W, C)
    image_tensor = torch.from_numpy(image_array)[None,]
    
    return image_tensor
    
def tensor_to_pil(tensor):
 
    # Handle batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert from torch tensor to numpy array
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Ensure values are in 0-255 range
    if tensor.dtype == np.float32 or tensor.dtype == np.float64:
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).astype(np.uint8)
        else:
            tensor = tensor.astype(np.uint8)
    
    # Convert to PIL Image
    if tensor.shape[-1] == 3:  # RGB
        return Image.fromarray(tensor, 'RGB')
    elif tensor.shape[-1] == 4:  # RGBA
        return Image.fromarray(tensor, 'RGBA')
    else:
        # Handle grayscale or other formats
        return Image.fromarray(tensor.squeeze(), 'L')

def create_mask_from_image(img):
    """Create a simple mask tensor from image dimensions"""
    return torch.ones((1, img.height, img.width), dtype=torch.float32)


def create_fallback_tensors(target_width, target_height, normalized=True):
    """Create fallback black image and mask tensors"""
    if normalized:
        fallback_img = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)
    else:
        fallback_img = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)
    
    fallback_mask = torch.zeros((1, target_height, target_width), dtype=torch.float32)
    return fallback_img, fallback_mask
    
def image_display(tensor, max_size=1024):
    
    arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.thumbnail((max_size, max_size))  # downscale
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
    
import numpy as np
import cv2

# ============================================================
# Low-Level Conversions
# ============================================================

def rgb_to_lab(rgb):
    rgb_uint8 = (np.clip(rgb * 255, 0, 255)).astype(np.uint8)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab

def lab_to_rgb(lab):
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return rgb

def rgb_to_xyz(rgb):
    def gamma_correct(c):
        return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))
    linear_rgb = gamma_correct(rgb)
    xyz = np.zeros_like(rgb)
    xyz[:,:,0] = 0.4124564 * linear_rgb[:,:,0] + 0.3575761 * linear_rgb[:,:,1] + 0.1804375 * linear_rgb[:,:,2]
    xyz[:,:,1] = 0.2126729 * linear_rgb[:,:,0] + 0.7151522 * linear_rgb[:,:,1] + 0.0721750 * linear_rgb[:,:,2]
    xyz[:,:,2] = 0.0193339 * linear_rgb[:,:,0] + 0.1191920 * linear_rgb[:,:,1] + 0.9503041 * linear_rgb[:,:,2]
    return xyz

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

def xyz_to_rgb(xyz):
    linear_rgb = np.zeros_like(xyz)
    linear_rgb[:,:,0] = 3.2404542 * xyz[:,:,0] - 1.5371385 * xyz[:,:,1] - 0.4985314 * xyz[:,:,2]
    linear_rgb[:,:,1] = -0.9692660 * xyz[:,:,0] + 1.8760108 * xyz[:,:,1] + 0.0415560 * xyz[:,:,2]
    linear_rgb[:,:,2] = 0.0556434 * xyz[:,:,0] - 0.2040259 * xyz[:,:,1] + 1.0572252 * xyz[:,:,2]
    def inverse_gamma_correct(c):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1/2.4) - 0.055)
    rgb = inverse_gamma_correct(linear_rgb)
    return np.clip(rgb, 0, 1)

# ============================================================
# Color Transfer Algorithms
# ============================================================

def reinhard_transfer(source, target):
    s_lab = self._rgb_to_lab(source)
    t_lab = self._rgb_to_lab(target)

    s_mu = np.mean(s_lab.reshape(-1, 3), axis=0)
    t_mu = np.mean(t_lab.reshape(-1, 3), axis=0)
    s_std = np.std(s_lab.reshape(-1, 3), axis=0) + 1e-6
    t_std = np.std(t_lab.reshape(-1, 3), axis=0)

    out = s_lab.copy()
    for i in range(3):
        out[:, :, i] = (s_lab[:, :, i] - s_mu[i]) * (t_std[i] / s_std[i]) + t_mu[i]
    return np.clip(self._lab_to_rgb(out), 0, 1)

def segment_based_transfer(source, target, k=8, attempts=3):
    """
    K-means clusters in Lab for source/target separately.
    Match source clusters to nearest target clusters by cluster mean in Lab.
    Apply per-cluster Reinhard stats using matched target pixels.
    """
    s_lab = self._rgb_to_lab(source)
    t_lab = self._rgb_to_lab(target)
    h, w, _ = s_lab.shape

    s_flat = s_lab.reshape(-1, 3).astype(np.float32)
    t_flat = t_lab.reshape(-1, 3).astype(np.float32)

    # OpenCV kmeans needs samples as N x features
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    K = max(2, min(k, len(s_flat)//500 if len(s_flat) > 1000 else 4))

    # Source clusters
    _, s_labels, s_centers = cv2.kmeans(s_flat, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    s_labels = s_labels.reshape(-1)

    # Target clusters
    _, t_labels, t_centers = cv2.kmeans(t_flat, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    t_labels = t_labels.reshape(-1)

    # Build per-cluster mapping by nearest center (Euclidean in Lab)
    # (Hungarian could be used, but nearest is fine and faster)
    matched_target_idx = []
    for c in s_centers:
        d = np.sum((t_centers - c)**2, axis=1)
        matched_target_idx.append(int(np.argmin(d)))
    matched_target_idx = np.array(matched_target_idx, dtype=np.int32)

    out_lab = s_lab.copy().reshape(-1, 3)

    for sc in range(K):
        s_mask = (s_labels == sc)
        if not np.any(s_mask):
            continue
        tc = matched_target_idx[sc]
        t_mask = (t_labels == tc)
        if not np.any(t_mask):
            # fallback to global stats
            s_mu = s_flat.mean(axis=0)
            s_std = s_flat.std(axis=0) + 1e-6
            t_mu = t_flat.mean(axis=0)
            t_std = t_flat.std(axis=0)
        else:
            s_mu = s_flat[s_mask].mean(axis=0)
            s_std = s_flat[s_mask].std(axis=0) + 1e-6
            t_mu = t_flat[t_mask].mean(axis=0)
            t_std = t_flat[t_mask].std(axis=0)

        out_lab[s_mask] = (s_flat[s_mask] - s_mu) * (t_std / s_std) + t_mu

    out_lab = out_lab.reshape(h, w, 3)
    return np.clip(self._lab_to_rgb(out_lab), 0, 1)

# ---- CIEDE2000 distance helpers ----
def _deltaE2000(Lab1, Lab2):
    """
    Lab1: [N,3], Lab2: [M,3] or [N,3] broadcasted pairwise if needed.
    Returns NxM matrix of ΔE00 if M>1; otherwise Nx1.
    Optimized for small M (we will subsample target).
    """
    # We compute pairwise distances Lab1 vs Lab2_sample
    L1, a1, b1 = Lab1[:, 0][:, None], Lab1[:, 1][:, None], Lab1[:, 2][:, None]
    L2, a2, b2 = Lab2[None, :, 0], Lab2[None, :, 1], Lab2[None, :, 2]

    # Implementation per Sharma et al. 2005
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt((C_bar**7) / (C_bar**7 + (25**7))))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    hp_bar = (h1p + h2p) / 2.0
    hp_bar = np.where(np.abs(h1p - h2p) > 180, hp_bar + 180, hp_bar)
    hp_bar = hp_bar % 360

    T = (1
         - 0.17 * np.cos(np.radians(hp_bar - 30))
         + 0.24 * np.cos(np.radians(2 * hp_bar))
         + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
         - 0.20 * np.cos(np.radians(4 * hp_bar - 63)))

    Sl = 1 + (0.015 * (Lp_bar - 50)**2) / np.sqrt(20 + (Lp_bar - 50)**2)
    Sc = 1 + 0.045 * Cp_bar
    Sh = 1 + 0.015 * Cp_bar * T

    Rt = -2 * np.sqrt((Cp_bar**7) / (Cp_bar**7 + 25**7)) * \
         np.sin(np.radians(60 * np.exp(-(((hp_bar - 275) / 25)**2))))

    dE = np.sqrt(
        (dLp / Sl)**2 +
        (dCp / Sc)**2 +
        (dHp / Sh)**2 +
        Rt * (dCp / Sc) * (dHp / Sh)
    )
    return dE  # shape [N, M]

def ciede2000_transfer(source, target, sample_size=600, chunk=2048):
    """
    For each source pixel (in chunks), find nearest target sample by ΔE00
    and map to that Lab value. Subsample target for speed.
    """
    s_lab = self._rgb_to_lab(source).reshape(-1, 3)
    t_lab = self._rgb_to_lab(target).reshape(-1, 3)

    M = min(sample_size, len(t_lab))
    # Random without replacement
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(t_lab), size=M, replace=False)
    t_sample = t_lab[idx]

    out = np.empty_like(s_lab)
    N = len(s_lab)
    for i in range(0, N, chunk):
        sl = s_lab[i:i+chunk]
        dE = self._deltaE2000(sl, t_sample)   # [chunk, M]
        nn = np.argmin(dE, axis=1)
        out[i:i+chunk] = t_sample[nn]

    out = out.reshape(source.shape)
    return np.clip(self._lab_to_rgb(out), 0, 1)

def hdr_color_transfer(source, target, tone_mapping='reinhard'):
    """HDR-aware global stats transfer with tone mapping."""
    source_is_hdr = np.any(source > 1.0)
    target_is_hdr = np.any(target > 1.0)

    def to_linear(img):
        return np.where(img <= 0.04045, img / 12.92, np.power((img + 0.055) / 1.055, 2.4))

    def from_linear(img):
        return np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(np.clip(img, 0, 1), 1/2.4) - 0.055)

    s_lin = to_linear(np.clip(source, 0, None)) if source_is_hdr else source
    t_lin = to_linear(np.clip(target, 0, None)) if target_is_hdr else target

    if tone_mapping == 'reinhard':
        def reinhard(img, wp=4.0):
            lum = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.0722*img[:, :, 2]
            lm = (lum / wp) / (1 + (lum / wp))
            scale = np.where(lum > 1e-8, lm / (lum + 1e-8), 1.0)
            return img * scale[:, :, None]
        if source_is_hdr: s_lin = reinhard(s_lin)
        if target_is_hdr: t_lin = reinhard(t_lin)

    elif tone_mapping == 'drago':
        def drago(img, bias=0.85):
            lum = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.0722*img[:, :, 2]
            logL = np.log10(lum + 1e-8)
            log_max, log_avg = np.max(logL), np.mean(logL)
            c1 = (bias * log_max) / (log_max - log_avg + 1e-8)
            c2 = (1 - bias) / (log_max - log_avg + 1e-8)
            lm = np.power(10, c1 + c2 * logL) / np.power(10, log_max + 1e-8)
            scale = np.where(lum > 1e-8, lm / (lum + 1e-8), 1.0)
            return img * scale[:, :, None]
        if source_is_hdr: s_lin = drago(s_lin)
        if target_is_hdr: t_lin = drago(t_lin)

    elif tone_mapping == 'linear_scale':
        if source_is_hdr: s_lin /= (np.max(s_lin) + 1e-8)
        if target_is_hdr: t_lin /= (np.max(t_lin) + 1e-8)

    # Global statistics match in Lab
    s_lab = self._rgb_to_lab(np.clip(s_lin, 0, 1))
    t_lab = self._rgb_to_lab(np.clip(t_lin, 0, 1))
    s_mu = np.mean(s_lab.reshape(-1, 3), axis=0)
    t_mu = np.mean(t_lab.reshape(-1, 3), axis=0)
    s_std = np.std(s_lab.reshape(-1, 3), axis=0) + 1e-6
    t_std = np.std(t_lab.reshape(-1, 3), axis=0)

    out = s_lab.copy()
    for i in range(3):
        out[:, :, i] = (s_lab[:, :, i] - s_mu[i]) * (t_std[i] / s_std[i]) + t_mu[i]

    rgb = self._lab_to_rgb(out)
    if source_is_hdr or target_is_hdr:
        rgb = from_linear(rgb)
    return np.clip(rgb, 0, 1)

# ---------- Regrain Postprocess ----------
def regrain(result_rgb, source_rgb, strength=0.6, radius=3):
    """
    Restore source high-frequency detail (grain/texture) in luminance.
    """
    # Work in YUV luminance
    src_yuv = self._rgb_to_yuv(source_rgb)
    res_yuv = self._rgb_to_yuv(result_rgb)

    src_y = src_yuv[:, :, 0]
    res_y = res_yuv[:, :, 0]

    # Detail = original - blurred(original)
    src_blur = cv2.GaussianBlur(src_y, (0, 0), sigmaX=radius, sigmaY=radius, borderType=cv2.BORDER_REPLICATE)
    detail = src_y - src_blur

    out_y = np.clip(res_y + strength * detail, 0, 1)
    out_yuv = res_yuv
    out_yuv[:, :, 0] = out_y
    return np.clip(self._yuv_to_rgb(out_yuv), 0, 1)

# ---------- Mixing ----------
def apply_single_method(source, target, method, iterations=3, gamut_mode='srgb', tone_mapping='reinhard'):
    if method == "reinhard":
        return self.reinhard_transfer(source, target)
    elif method == "segment_based":
        # iterations loosely controls k here
        k = 4 + 2 * max(0, int(iterations) - 1)
        return self.segment_based_transfer(source, target, k=k)
    elif method == "ciede2000_transfer":
        # iterations controls passes/chunking indirectly
        return self.ciede2000_transfer(source, target, sample_size=600, chunk=2048)
    elif method == "hdr_color_transfer":
        return self.hdr_color_transfer(source, target, tone_mapping)
    else:
        # Fallback
        return self.reinhard_transfer(source, target)

def blend_mixing(result1, result2, weight):
    return weight * result1 + (1 - weight) * result2

def sequential_mixing(source, target, primary_method, secondary_method, weight, iterations, gamut_mode, tone_mapping):
    inter = self.apply_single_method(source, target, primary_method, iterations, gamut_mode, tone_mapping)
    fin = self.apply_single_method(inter, target, secondary_method, iterations, gamut_mode, tone_mapping)
    return weight * inter + (1 - weight) * fin

def frequency_split_mixing(source, target, primary_method, secondary_method, weight, gamut_mode, tone_mapping):
    def gauss_pyr(image, levels=3):
        pyr = [image]
        for _ in range(levels): pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr
    def lap_pyr(gp):
        lp = []
        for i in range(len(gp) - 1):
            up = cv2.pyrUp(gp[i + 1])
            if up.shape[:2] != gp[i].shape[:2]:
                up = cv2.resize(up, (gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp
    def reconstruct(lp):
        res = lp[-1]
        for i in range(len(lp) - 2, -1, -1):
            res = cv2.pyrUp(res)
            if res.shape[:2] != lp[i].shape[:2]:
                res = cv2.resize(res, (lp[i].shape[1], lp[i].shape[0]))
            res = res + lp[i]
        return res

    low = self.apply_single_method(source, target, primary_method, 5, gamut_mode, tone_mapping)
    high = self.apply_single_method(source, target, secondary_method, 5, gamut_mode, tone_mapping)
    low_p = lap_pyr(gauss_pyr(low))
    high_p = lap_pyr(gauss_pyr(high))
    mixed = []
    for i, (l, h) in enumerate(zip(low_p, high_p)):
        if i < len(low_p)//2:
            mixed.append(weight * l + (1 - weight) * h)
        else:
            mixed.append((1 - weight) * l + weight * h)
    return np.clip(reconstruct(mixed), 0, 1)

def luminance_split_mixing(source, target, primary_method, secondary_method, weight, gamut_mode, tone_mapping):
    yuv = self._rgb_to_yuv(source)
    Y = yuv[:, :, 0]
    bright = (Y > np.mean(Y)).astype(np.float32)
    dark = 1.0 - bright

    r1 = self.apply_single_method(source, target, primary_method, 5, gamut_mode, tone_mapping)
    r2 = self.apply_single_method(source, target, secondary_method, 5, gamut_mode, tone_mapping)

    bright_blend = weight * r1 + (1 - weight) * r2
    dark_blend = (1 - weight) * r1 + weight * r2
    res = bright[:, :, None] * bright_blend + dark[:, :, None] * dark_blend
    return np.clip(res, 0, 1)

def detail_preserve_mixing(source, target, primary_method, secondary_method, weight, gamut_mode, tone_mapping):
    src_gray = cv2.cvtColor((np.clip(source,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    details = src_gray - cv2.GaussianBlur(src_gray, (5, 5), 1.0)
    r1 = self.apply_single_method(source, target, primary_method, 5, gamut_mode, tone_mapping)
    r2 = self.apply_single_method(source, target, secondary_method, 5, gamut_mode, tone_mapping)
    blended = weight * r1 + (1 - weight) * r2
    bgray = cv2.cvtColor((np.clip(blended,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    egray = np.clip(bgray + details * 0.5, 0, 1)
    byuv = self._rgb_to_yuv(blended)
    byuv[:, :, 0] = egray
    return self._yuv_to_rgb(byuv)
