import torch
import numpy as np
import cv2
from scipy import stats, ndimage, spatial
from scipy.stats import wasserstein_distance
# Assuming these are part of your project structure
# from ...core.node import node_path, node_prefix, main_cetegory

class MixedColorTransfer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "primary_method": ([
                    "reinhard", "mkl", "hm_mvgd", "linear_stats", "lalphabeta", "wavelet_color",
                    "pdf_transfer", "iterative_distribution", "gradual_blending", "regrain",
                    "color_mood", "neural_style_color", "segment_based", "adaptive_manifold",
                    "ciede2000_transfer", "gamut_mapping", "hdr_color_transfer"
                ],),
                "secondary_method": ([
                    "none", "reinhard", "mkl", "hm_mvgd", "linear_stats", "lalphabeta", "wavelet_color",
                    "pdf_transfer", "iterative_distribution", "gradual_blending", "regrain",
                    "color_mood", "neural_style_color", "segment_based", "adaptive_manifold",
                    "ciede2000_transfer", "gamut_mapping", "hdr_color_transfer"
                ],),
                "mixing_mode": (["blend", "sequential", "frequency_split", "luminance_split", "detail_preserve"],),
                "primary_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "preserve_luminance": ("BOOLEAN", {"default": True}),
                "iterations": ("INT", {"default": 5, "min": 1, "max": 20}),
                "gamut_mode": (["srgb", "adobe_rgb", "prophoto_rgb", "rec2020", "dci_p3", "auto_detect"],),
                "hdr_tone_mapping": (["reinhard", "drago", "mantiuk", "linear_scale"],),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mixed_color_transfer"
    CATEGORY = "Artha/Color/Mixed"

    def __init__(self):
        # Initialize method classes and link them back to this parent instance
        self.advanced_methods = self._init_advanced_methods()
        self.advanced_methods.parent = self
        self.additional_methods = self._init_additional_methods()
        self.additional_methods.parent = self
        self.perceptual_methods = self._init_perceptual_methods()
        self.perceptual_methods.parent = self

    # --- Consolidated Helper Methods ---
    def _rgb_to_lab(self, rgb):
        rgb_uint8 = (np.clip(rgb * 255, 0, 255)).astype(np.uint8)
        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        return lab

    def _lab_to_rgb(self, lab):
        lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return rgb

    # --- Method Initializers ---
    def _init_advanced_methods(self):
        """Initialize methods from AdvancedColorMatching"""
        class AdvancedMethods:
            def reinhard_color_transfer(self, source, target):
                source_lab = self.parent._rgb_to_lab(source)
                target_lab = self.parent._rgb_to_lab(target)

                source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
                source_std = np.std(source_lab.reshape(-1, 3), axis=0)
                target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
                target_std = np.std(target_lab.reshape(-1, 3), axis=0)

                result_lab = source_lab.copy()
                
                for i in range(3):
                    if source_std[i] > 1e-6:
                        result_lab[:, :, i] = (source_lab[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]

                return self.parent._lab_to_rgb(result_lab)

            def mkl_color_transfer(self, source, target):
                source_lab = self.parent._rgb_to_lab(source).reshape(-1, 3)
                target_lab = self.parent._rgb_to_lab(target).reshape(-1, 3)

                source_cov = np.cov(source_lab.T)
                target_cov = np.cov(target_lab.T)
                source_mean = np.mean(source_lab, axis=0)
                target_mean = np.mean(target_lab, axis=0)

                try:
                    source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
                    source_sqrt = source_eigvecs @ np.diag(np.sqrt(np.maximum(source_eigvals, 1e-8))) @ source_eigvecs.T
                    source_inv_sqrt = source_eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(source_eigvals, 1e-8))) @ source_eigvecs.T

                    target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)
                    target_sqrt = target_eigvecs @ np.diag(np.sqrt(np.maximum(target_eigvals, 1e-8))) @ target_eigvecs.T

                    T = source_inv_sqrt @ target_sqrt @ source_inv_sqrt @ source_sqrt
                    result_lab = ((source_lab - source_mean) @ T.T + target_mean).reshape(source.shape[:2] + (3,))

                except np.linalg.LinAlgError:
                    return self.reinhard_color_transfer(source, target)

                return self.parent._lab_to_rgb(result_lab)
            # Add other methods from AdvancedColorMatching as needed...
        return AdvancedMethods()

    def _init_additional_methods(self):
        """Initialize methods from AdditionalColorMethods"""
        class AdditionalMethods:
            def pdf_transfer_method(self, source, target):
                source_lab = self.parent._rgb_to_lab(source).reshape(-1, 3)
                target_lab = self.parent._rgb_to_lab(target).reshape(-1, 3)
                result_lab = np.zeros_like(source_lab)

                for channel in range(3):
                    source_channel = source_lab[:, channel]
                    target_channel = target_lab[:, channel]
                    n_bins = min(256, len(np.unique(source_channel)))

                    source_hist, source_bin_edges = np.histogram(source_channel, bins=n_bins, density=True)
                    source_cdf = np.cumsum(source_hist) * (source_bin_edges[1] - source_bin_edges[0])
                    source_bin_centers = (source_bin_edges[:-1] + source_bin_edges[1:]) / 2

                    target_hist, target_bin_edges = np.histogram(target_channel, bins=n_bins, density=True)
                    target_cdf = np.cumsum(target_hist) * (target_bin_edges[1] - target_bin_edges[0])
                    target_bin_centers = (target_bin_edges[:-1] + target_bin_edges[1:]) / 2

                    source_cdf = source_cdf / source_cdf[-1] if source_cdf[-1] > 0 else source_cdf
                    target_cdf = target_cdf / target_cdf[-1] if target_cdf[-1] > 0 else target_cdf

                    source_cdf_values = np.interp(source_channel, source_bin_centers, source_cdf)
                    result_lab[:, channel] = np.interp(source_cdf_values, target_cdf, target_bin_centers)

                return self.parent._lab_to_rgb(result_lab.reshape(source.shape[:2] + (3,)))

            def color_mood_transfer(self, source, target):
                source_hsv = cv2.cvtColor((source * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                target_hsv = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                result_hsv = source_hsv.copy()
                target_hues = target_hsv[:, :, 0].flatten()
                target_sats = target_hsv[:, :, 1].flatten()

                hue_hist, _ = np.histogram(target_hues, bins=36, range=(0, 180), weights=target_sats)
                dominant_hue = np.argmax(hue_hist) * 5

                source_hues = source_hsv[:, :, 0]
                hue_diff = dominant_hue - np.mean(source_hues)
                result_hsv[:, :, 0] = np.mod(source_hues + hue_diff * 0.3, 180)

                source_sat_mean = np.mean(source_hsv[:, :, 1])
                target_sat_mean = np.mean(target_hsv[:, :, 1])
                sat_ratio = target_sat_mean / (source_sat_mean + 1e-8)
                result_hsv[:, :, 1] = np.clip(source_hsv[:, :, 1] * sat_ratio, 0, 255)

                result_rgb = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
                return result_rgb
            # Add other methods from AdditionalColorMethods as needed...
        return AdditionalMethods()

    def _init_perceptual_methods(self):
        """Initialize new perceptual methods"""
        class PerceptualMethods:
            def rgb_to_xyz(self, rgb):
                """Convert RGB to XYZ color space"""
                def gamma_correct(c):
                    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))
                linear_rgb = gamma_correct(rgb)
                xyz = np.zeros_like(rgb)
                xyz[:,:,0] = 0.4124564 * linear_rgb[:,:,0] + 0.3575761 * linear_rgb[:,:,1] + 0.1804375 * linear_rgb[:,:,2]
                xyz[:,:,1] = 0.2126729 * linear_rgb[:,:,0] + 0.7151522 * linear_rgb[:,:,1] + 0.0721750 * linear_rgb[:,:,2]
                xyz[:,:,2] = 0.0193339 * linear_rgb[:,:,0] + 0.1191920 * linear_rgb[:,:,1] + 0.9503041 * linear_rgb[:,:,2]
                return xyz

            def xyz_to_lab(self, xyz):
                """Convert XYZ to LAB color space"""
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

            def ciede2000_color_transfer(self, source, target):
                """CIEDE2000-based perceptually uniform color transfer"""
                source_xyz = self.rgb_to_xyz(source)
                target_xyz = self.rgb_to_xyz(target)
                source_lab = self.xyz_to_lab(source_xyz)
                target_lab = self.xyz_to_lab(target_xyz)
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
                result_xyz = self.lab_to_xyz(result_lab)
                result_rgb = self.xyz_to_rgb(result_xyz)
                return np.clip(result_rgb, 0, 1)

            def lab_to_xyz(self, lab):
                """Convert LAB to XYZ color space"""
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

            def xyz_to_rgb(self, xyz):
                """Convert XYZ to RGB color space"""
                linear_rgb = np.zeros_like(xyz)
                linear_rgb[:,:,0] = 3.2404542 * xyz[:,:,0] - 1.5371385 * xyz[:,:,1] - 0.4985314 * xyz[:,:,2]
                linear_rgb[:,:,1] = -0.9692660 * xyz[:,:,0] + 1.8760108 * xyz[:,:,1] + 0.0415560 * xyz[:,:,2]
                linear_rgb[:,:,2] = 0.0556434 * xyz[:,:,0] - 0.2040259 * xyz[:,:,1] + 1.0572252 * xyz[:,:,2]
                def inverse_gamma_correct(c):
                    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1/2.4) - 0.055)
                rgb = inverse_gamma_correct(linear_rgb)
                return np.clip(rgb, 0, 1)

            def get_gamut_matrix(self, gamut_name):
                """Get color gamut transformation matrices"""
                gamuts = {
                    'srgb': {'primaries': [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], 'white': [0.3127, 0.3290]},
                    'adobe_rgb': {'primaries': [[0.64, 0.33], [0.21, 0.71], [0.15, 0.06]], 'white': [0.3127, 0.3290]},
                    'prophoto_rgb': {'primaries': [[0.7347, 0.2653], [0.1596, 0.8404], [0.0366, 0.0001]], 'white': [0.3457, 0.3585]},
                    'rec2020': {'primaries': [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], 'white': [0.3127, 0.3290]},
                    'dci_p3': {'primaries': [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]], 'white': [0.314, 0.351]}
                }
                return gamuts.get(gamut_name.lower(), gamuts['srgb'])

            def gamut_mapping_transfer(self, source, target, gamut_mode='srgb'):
                """Gamut-aware color transfer"""
                if gamut_mode == 'auto_detect':
                    max_vals = np.max(source.reshape(-1, 3), axis=0)
                    if np.any(max_vals > 0.95): gamut_mode = 'rec2020'
                    elif np.mean(max_vals) > 0.8: gamut_mode = 'adobe_rgb'
                    else: gamut_mode = 'srgb'
                gamut_info = self.get_gamut_matrix(gamut_mode)
                source_lab = self.parent._rgb_to_lab(source)
                target_lab = self.parent._rgb_to_lab(target)
                gamut_radius = 128 if gamut_mode in ['rec2020', 'prophoto_rgb'] else 100
                source_mean, target_mean = np.mean(source_lab.reshape(-1, 3), axis=0), np.mean(target_lab.reshape(-1, 3), axis=0)
                source_std, target_std = np.std(source_lab.reshape(-1, 3), axis=0), np.std(target_lab.reshape(-1, 3), axis=0)
                result_lab = source_lab.copy()
                for i in range(3):
                    if source_std[i] > 1e-6:
                        result_lab[:, :, i] = ((source_lab[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i])
                a_channel, b_channel = result_lab[:, :, 1], result_lab[:, :, 2]
                chroma_distance = np.sqrt(a_channel**2 + b_channel**2)
                outside_gamut = chroma_distance > gamut_radius
                if np.any(outside_gamut):
                    scale_factor = gamut_radius / (chroma_distance + 1e-8)
                    result_lab[:, :, 1] = np.where(outside_gamut, a_channel * scale_factor, a_channel)
                    result_lab[:, :, 2] = np.where(outside_gamut, b_channel * scale_factor, b_channel)
                return self.parent._lab_to_rgb(result_lab)

            def hdr_color_transfer(self, source, target, tone_mapping='reinhard'):
                """HDR-aware color transfer with tone mapping"""
                source_is_hdr = np.any(source > 1.0)
                target_is_hdr = np.any(target > 1.0)
                def to_linear(img): return np.where(img <= 0.04045, img / 12.92, np.power((img + 0.055) / 1.055, 2.4))
                def from_linear(img): return np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(np.clip(img, 0, 1), 1/2.4) - 0.055)
                
                source_linear = to_linear(np.clip(source, 0, None)) if source_is_hdr else source
                target_linear = to_linear(np.clip(target, 0, None)) if target_is_hdr else target

                if tone_mapping == 'reinhard':
                    def reinhard_tonemap(img, white_point=4.0):
                        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
                        lum_mapped = (lum / white_point) / (1 + (lum / white_point))
                        scale = np.where(lum > 1e-8, lum_mapped / (lum + 1e-8), 1.0)
                        return img * scale[:, :, np.newaxis]
                    if source_is_hdr: source_linear = reinhard_tonemap(source_linear)
                    if target_is_hdr: target_linear = reinhard_tonemap(target_linear)
                
                elif tone_mapping == 'drago':
                    def drago_tonemap(img, bias=0.85):
                        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
                        log_lum_max, log_lum_avg = np.max(np.log10(lum + 1e-8)), np.mean(np.log10(lum + 1e-8))
                        c1 = (bias * log_lum_max) / (log_lum_max - log_lum_avg)
                        c2 = (1 - bias) / (log_lum_max - log_lum_avg)
                        lum_mapped = np.power(10, c1 + c2 * np.log10(lum + 1e-8)) / np.power(10, log_lum_max)
                        scale = np.where(lum > 1e-8, lum_mapped / (lum + 1e-8), 1.0)
                        return img * scale[:, :, np.newaxis]
                    if source_is_hdr: source_linear = drago_tonemap(source_linear)
                    if target_is_hdr: target_linear = drago_tonemap(target_linear)
                
                elif tone_mapping == 'linear_scale':
                    if source_is_hdr: source_linear /= (np.max(source_linear) + 1e-8)
                    if target_is_hdr: target_linear /= (np.max(target_linear) + 1e-8)
                
                source_lab = self.parent._rgb_to_lab(np.clip(source_linear, 0, 1))
                target_lab = self.parent._rgb_to_lab(np.clip(target_linear, 0, 1))
                source_mean, target_mean = np.mean(source_lab.reshape(-1, 3), axis=0), np.mean(target_lab.reshape(-1, 3), axis=0)
                source_std, target_std = np.std(source_lab.reshape(-1, 3), axis=0), np.std(target_lab.reshape(-1, 3), axis=0)
                result_lab = source_lab.copy()
                for i in range(3):
                    if source_std[i] > 1e-6:
                        result_lab[:, :, i] = ((source_lab[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i])
                result = self.parent._lab_to_rgb(result_lab)
                if source_is_hdr or target_is_hdr: result = from_linear(result)
                return np.clip(result, 0, 1)
        return PerceptualMethods()

    # --- Main Logic ---
    def apply_single_method(self, source, target, method, iterations=5, gamut_mode='srgb', tone_mapping='reinhard'):
        """Apply a single color transfer method"""
        if method == "reinhard":
            return self.advanced_methods.reinhard_color_transfer(source, target)
        elif method == "mkl":
            return self.advanced_methods.mkl_color_transfer(source, target)
        elif method == "pdf_transfer":
            return self.additional_methods.pdf_transfer_method(source, target)
        elif method == "color_mood":
            return self.additional_methods.color_mood_transfer(source, target)
        elif method == "ciede2000_transfer":
            return self.perceptual_methods.ciede2000_color_transfer(source, target)
        elif method == "gamut_mapping":
            return self.perceptual_methods.gamut_mapping_transfer(source, target, gamut_mode)
        elif method == "hdr_color_transfer":
            return self.perceptual_methods.hdr_color_transfer(source, target, tone_mapping)
        else: # Default to reinhard if method not implemented
            return self.advanced_methods.reinhard_color_transfer(source, target)

    def blend_mixing(self, result1, result2, weight):
        """Simple weighted blending of two results"""
        return weight * result1 + (1 - weight) * result2

    def sequential_mixing(self, source, target, primary_method, secondary_method, weight, iterations, gamut_mode, tone_mapping):
        """Apply methods sequentially"""
        intermediate = self.apply_single_method(source, target, primary_method, iterations, gamut_mode, tone_mapping)
        final = self.apply_single_method(intermediate, target, secondary_method, iterations, gamut_mode, tone_mapping)
        return weight * intermediate + (1 - weight) * final

    def frequency_split_mixing(self, source, target, primary_method, secondary_method, weight, gamut_mode, tone_mapping):
        """Apply different methods to different frequency components"""
        def gaussian_pyramid(image, levels=3):
            pyramid = [image]
            for i in range(levels): pyramid.append(cv2.pyrDown(pyramid[-1]))
            return pyramid
        def laplacian_pyramid(gaussian_pyr):
            laplacian_pyr = []
            for i in range(len(gaussian_pyr) - 1):
                expanded = cv2.pyrUp(gaussian_pyr[i + 1])
                if expanded.shape[:2] != gaussian_pyr[i].shape[:2]:
                    expanded = cv2.resize(expanded, (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
                laplacian_pyr.append(gaussian_pyr[i] - expanded)
            laplacian_pyr.append(gaussian_pyr[-1])
            return laplacian_pyr
        def reconstruct_from_laplacian(laplacian_pyr):
            result = laplacian_pyr[-1]
            for i in range(len(laplacian_pyr) - 2, -1, -1):
                result = cv2.pyrUp(result)
                if result.shape[:2] != laplacian_pyr[i].shape[:2]:
                    result = cv2.resize(result, (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0]))
                result += laplacian_pyr[i]
            return result
        
        low_freq_result = self.apply_single_method(source, target, primary_method, 5, gamut_mode, tone_mapping)
        high_freq_result = self.apply_single_method(source, target, secondary_method, 5, gamut_mode, tone_mapping)
        low_pyr = laplacian_pyramid(gaussian_pyramid(low_freq_result))
        high_pyr = laplacian_pyramid(gaussian_pyramid(high_freq_result))
        mixed_pyr = []
        for i, (low_level, high_level) in enumerate(zip(low_pyr, high_pyr)):
            if i < len(low_pyr) // 2: mixed_pyr.append(weight * low_level + (1 - weight) * high_level)
            else: mixed_pyr.append((1 - weight) * low_level + weight * high_level)
        return np.clip(reconstruct_from_laplacian(mixed_pyr), 0, 1)

    def luminance_split_mixing(self, source, target, primary_method, secondary_method, weight, gamut_mode, tone_mapping):
        """Apply different methods based on luminance"""
        source_yuv = cv2.cvtColor((source * 255).astype(np.uint8), cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        luminance = source_yuv[:, :, 0]
        bright_mask = (luminance > np.mean(luminance)).astype(np.float32)
        dark_mask = 1.0 - bright_mask
        result1 = self.apply_single_method(source, target, primary_method, 5, gamut_mode, tone_mapping)
        result2 = self.apply_single_method(source, target, secondary_method, 5, gamut_mode, tone_mapping)
        bright_blend = weight * result1 + (1 - weight) * result2
        dark_blend = (1 - weight) * result1 + weight * result2
        result = (bright_mask[:, :, np.newaxis] * bright_blend + dark_mask[:, :, np.newaxis] * dark_blend)
        return np.clip(result, 0, 1)

    def detail_preserve_mixing(self, source, target, primary_method, secondary_method, weight, gamut_mode, tone_mapping):
        """Mix methods while preserving fine details"""
        source_gray = cv2.cvtColor((source * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        details = source_gray - cv2.GaussianBlur(source_gray, (5, 5), 1.0)
        result1 = self.apply_single_method(source, target, primary_method, 5, gamut_mode, tone_mapping)
        result2 = self.apply_single_method(source, target, secondary_method, 5, gamut_mode, tone_mapping)
        blended = weight * result1 + (1 - weight) * result2
        blended_gray = cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        enhanced_gray = np.clip(blended_gray + details * 0.5, 0, 1)
        blended_yuv = cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        blended_yuv[:, :, 0] = enhanced_gray
        result = cv2.cvtColor((np.clip(blended_yuv, 0, 1) * 255).astype(np.uint8), cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
        return result

    def mixed_color_transfer(self, source_image, target_image, primary_method, secondary_method,
                           mixing_mode, primary_weight, preserve_luminance, iterations, strength,
                           gamut_mode, hdr_tone_mapping):
        source = source_image[0].cpu().numpy()
        target = target_image[0].cpu().numpy()

        if preserve_luminance:
            source_gray = np.mean(source, axis=2, keepdims=True)

        if secondary_method == "none":
            result = self.apply_single_method(source, target, primary_method, iterations, gamut_mode, hdr_tone_mapping)
        else:
            mixing_map = {
                "blend": lambda: self.blend_mixing(
                    self.apply_single_method(source, target, primary_method, iterations, gamut_mode, hdr_tone_mapping),
                    self.apply_single_method(source, target, secondary_method, iterations, gamut_mode, hdr_tone_mapping),
                    primary_weight
                ),
                "sequential": lambda: self.sequential_mixing(source, target, primary_method, secondary_method, primary_weight, iterations, gamut_mode, hdr_tone_mapping),
                "frequency_split": lambda: self.frequency_split_mixing(source, target, primary_method, secondary_method, primary_weight, gamut_mode, hdr_tone_mapping),
                "luminance_split": lambda: self.luminance_split_mixing(source, target, primary_method, secondary_method, primary_weight, gamut_mode, hdr_tone_mapping),
                "detail_preserve": lambda: self.detail_preserve_mixing(source, target, primary_method, secondary_method, primary_weight, gamut_mode, hdr_tone_mapping),
            }
            result = mixing_map.get(mixing_mode, mixing_map["blend"])()
            
        if preserve_luminance:
            result_gray = np.mean(result, axis=2, keepdims=True)
            result = result * (source_gray / (result_gray + 1e-8))

        result = strength * result + (1 - strength) * source
        result = np.clip(result, 0, 1)
        result_tensor = torch.from_numpy(result).unsqueeze(0).float()
        return (result_tensor,)

# Mock missing variables for standalone execution
def node_prefix(): return "TestPrefix"

NODE_CLASS_MAPPINGS = {
    "MixedColorTransfer": MixedColorTransfer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MixedColorTransfer": node_prefix() + " MIXED COLOR TRANSFER",
}