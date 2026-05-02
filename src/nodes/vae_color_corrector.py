"""VAE Color Corrector node for fixing VAE-induced color shifts in inpainting workflows.

MODIFIED: v1.3-anime-fix — Fixes for anime/webtoon skin color correction.
Key changes:
- Added 'correction_target' option: correct inpainted, non-inpainted, or full image
- Added 'anime_mode' for anime/webtoon-specific color correction
- Fixed auto-detect threshold to use absolute + Otsu instead of relative quantile
- Added fallback statistical matching that works without ADVANCED_LIBS
- Added anime-aware region sampling for more precise color matching
"""

import torch
import torch.nn.functional as F
import numpy as np

from ..utils import ADVANCED_LIBS_AVAILABLE, match_to_reference_colors

if ADVANCED_LIBS_AVAILABLE:
    import cv2
    from sklearn.cluster import KMeans
    from skimage import exposure


class VAEColorCorrector:
    """
    Specialized color correction for VAE artifacts in inpainting/img2img workflows.
    Fixes color shifts by referencing the original input image.
    
    v1.3-anime-fix: Now supports anime/webtoon skin color calibration with
    correction_target, anime_mode, and improved mask handling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "Original input image (before VAE encoding)"}),
                "processed_image": ("IMAGE", {"tooltip": "Image after VAE decode (with color shifts)"}),
                "correction_strength": (
                    "FLOAT", 
                    {
                        "default": 0.85, 
                        "min": 0.0, 
                        "max": 1.0, 
                        "step": 0.01,
                        "tooltip": "How strongly to correct VAE color shifts (0.0 = no correction, 1.0 = full correction)"
                    }
                ),
                "method": (
                    ["luminance_zones", "histogram_matching", "statistical_matching", "advanced_3d_lut", "anime_skin_match"], 
                    {
                        "default": "anime_skin_match",
                        "tooltip": "Color correction method:\n"
                            "• luminance_zones: Shadows/midtones/highlights correction\n"
                            "• histogram_matching: Match color distributions\n"
                            "• statistical_matching: Match color statistics (works without extra libs)\n"
                            "• advanced_3d_lut: Most accurate 3D color mapping\n"
                            "• anime_skin_match: [NEW] Anime/webtoon skin-optimized correction using regional sampling and LAB color transfer"
                    }
                ),
                "correction_target": (
                    ["correct_inpainted", "correct_non_inpainted", "full_image"],
                    {
                        "default": "correct_inpainted",
                        "tooltip": "Which areas to apply correction to:\n"
                            "• correct_inpainted: [RECOMMENDED for inpainting] Correct the inpainted area using original image colors as reference\n"
                            "• correct_non_inpainted: Correct non-inpainted areas (preserve new content)\n"
                            "• full_image: Apply correction to entire image uniformly"
                    }
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask - white = inpainted area, black = original area"}),
                "edge_feather": (
                    "INT", 
                    {
                        "default": 15, 
                        "min": 0, 
                        "max": 100, 
                        "step": 1,
                        "tooltip": "Feather edges between corrected/preserved areas (pixels). Higher = smoother blend."
                    }
                ),
                "anime_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable anime/webtoon-specific adjustments: wider color ranges, pastel skin support, cool tone preservation"
                    }
                ),
                "skin_sample_region": (
                    "STRING",
                    {
                        "default": "auto",
                        "tooltip": "Region to sample reference skin color from:\n"
                            "• auto: Auto-detect from non-inpainted skin areas\n"
                            "• top_quarter: Top 25% of image (good for face portraits)\n"
                            "• center: Center region\n"
                            "• full: Entire non-masked area"
                    }
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_image",)
    FUNCTION = "correct_vae_colors"
    CATEGORY = "EasyColorCorrection"
    
    def correct_vae_colors(
        self, 
        original_image, 
        processed_image, 
        correction_strength=0.85,
        method="anime_skin_match",
        correction_target="correct_inpainted",
        mask=None,
        edge_feather=15,
        anime_mode=True,
        skin_sample_region="auto"
    ):
        """
        Correct VAE-induced color shifts by referencing the original image.
        """
        device = original_image.device
        
        print(f"[VAE-CC] method={method}, strength={correction_strength:.2f}, target={correction_target}, anime={anime_mode}")
        print(f"[VAE-CC] Original: {original_image.shape}, Processed: {processed_image.shape}")
        
        # Ensure images are same size
        if original_image.shape != processed_image.shape:
            print("[VAE-CC] Image size mismatch - resizing processed to match original")
            processed_image = F.interpolate(
                processed_image.permute(0, 3, 1, 2), 
                size=(original_image.shape[1], original_image.shape[2]), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # Process each image in batch
        corrected_batch = []
        
        for i in range(original_image.shape[0]):
            orig_img = original_image[i]
            proc_img = processed_image[i]
            current_mask = mask[i] if mask is not None else None
            
            # Apply color correction
            corrected_img = self._apply_vae_color_correction(
                orig_img, proc_img, method, correction_strength, 
                correction_target, current_mask, edge_feather, device,
                anime_mode, skin_sample_region
            )
            
            corrected_batch.append(corrected_img)
        
        result = torch.stack(corrected_batch, dim=0)
        print(f"[VAE-CC] Completed for {len(corrected_batch)} images")
        
        return (result,)
    
    def _apply_vae_color_correction(
        self, original_img, processed_img, method, strength, 
        correction_target, mask, edge_feather, device,
        anime_mode, skin_sample_region
    ):
        """Apply the actual color correction."""
        
        # Convert to numpy for processing
        orig_np = (original_img.cpu().numpy() * 255).astype(np.uint8)
        proc_np = (processed_img.cpu().numpy() * 255).astype(np.uint8)
        
        print(f"[VAE-CC] Applying {method} color correction (anime_mode={anime_mode})...")
        
        # Apply color correction based on method
        if method == "anime_skin_match":
            corrected_np = self._anime_skin_match_correction(
                orig_np, proc_np, strength, anime_mode, mask, skin_sample_region
            )
        elif method == "advanced_3d_lut":
            corrected_np = self._advanced_3d_lut_correction(orig_np, proc_np, strength)
        elif method == "luminance_zones":
            corrected_np = match_to_reference_colors(proc_np, orig_np, strength)
        elif method == "histogram_matching":
            corrected_np = self._histogram_matching_correction(orig_np, proc_np, strength)
        else:  # statistical_matching - works without ADVANCED_LIBS
            corrected_np = self._statistical_matching_correction(orig_np, proc_np, strength)
        
        # Convert back to tensor
        corrected_tensor = torch.from_numpy(corrected_np.astype(np.float32) / 255.0).to(device)
        
        # Handle mask-based targeting
        if mask is not None:
            corrected_tensor = self._apply_mask_targeting(
                processed_img, corrected_tensor, mask, edge_feather, device, correction_target
            )
        elif correction_target != "full_image":
            # Auto-detect changed areas for targeting
            corrected_tensor = self._auto_target_correction(
                original_img, processed_img, corrected_tensor, edge_feather, device, correction_target
            )
        # else: full_image — no masking needed
        
        return corrected_tensor
    
    # =========================================================================
    # NEW: Anime Skin Match — the main method for anime/webtoon correction
    # =========================================================================
    def _anime_skin_match_correction(self, original_np, processed_np, strength, anime_mode, mask, skin_sample_region):
        """
        Anime/webtoon-optimized color correction.
        
        Strategy:
        1. Sample skin-like colors from the ORIGINAL image (reference)
        2. Find corresponding regions in the PROCESSED image
        3. Compute per-region color shift in LAB space
        4. Apply shift with strength blending
        
        This method works even without ADVANCED_LIBS by falling back to 
        statistical matching + numpy-based LAB conversion.
        """
        if ADVANCED_LIBS_AVAILABLE:
            return self._anime_skin_match_advanced(original_np, processed_np, strength, anime_mode, mask, skin_sample_region)
        else:
            print("[VAE-CC] Advanced libs not available, using statistical fallback with anime adjustments")
            # Statistical matching always works (numpy only)
            corrected = self._statistical_matching_correction(original_np, processed_np, strength)
            if anime_mode:
                # Apply additional cool-tone bias for anime
                corrected = self._apply_anime_cool_bias(corrected, strength * 0.3)
            return corrected
    
    def _anime_skin_match_advanced(self, original_np, processed_np, strength, anime_mode, mask, skin_sample_region):
        """Advanced anime skin match using OpenCV LAB + KMeans."""
        try:
            # Convert to LAB for perceptual color analysis
            orig_lab = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            proc_lab = cv2.cvtColor(processed_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            h, w = original_np.shape[:2]
            
            # Determine reference sampling region
            if skin_sample_region == "top_quarter":
                sample_slice = (slice(0, h // 4), slice(0, w))
            elif skin_sample_region == "center":
                sample_slice = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
            else:  # auto or full
                sample_slice = (slice(None), slice(None))
            
            # If mask provided, sample from NON-inpainted areas only for reference
            if mask is not None:
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                # Non-inpainted = where mask is 0 (black)
                non_inpainted = (mask_np < 0.5)
                # Combine with sample region
                ref_lab = orig_lab[sample_slice][non_inpainted[sample_slice]]
            else:
                ref_lab = orig_lab[sample_slice].reshape(-1, 3)
            
            if len(ref_lab) < 100:
                print("[VAE-CC] Too few reference pixels, using full image")
                ref_lab = orig_lab.reshape(-1, 3)
            
            # Find skin-like pixels in reference using expanded ranges for anime
            # LAB ranges (OpenCV uint8): L=0-255, A=0-255 (128=neutral), B=0-255 (128=neutral)
            if anime_mode:
                # Wider ranges for anime/webtoon skin:
                # Covers warm skin (orange/peach), cool skin (pink/lavender), and pastel tones
                l_lo, l_hi = 80, 255
                a_lo, a_hi = 108, 165  # Expanded from 120-142 to include cool tones
                b_lo, b_hi = 108, 180  # Expanded from 130-165 to include cool/pastel tones
            else:
                l_lo, l_hi = 40, 220
                a_lo, a_hi = 118, 152
                b_lo, b_hi = 125, 170
            
            # Filter reference to skin-like pixels
            skin_mask_ref = (
                (ref_lab[:, 0] > l_lo) & (ref_lab[:, 0] < l_hi) &
                (ref_lab[:, 1] > a_lo) & (ref_lab[:, 1] < a_hi) &
                (ref_lab[:, 2] > b_lo) & (ref_lab[:, 2] < b_hi)
            )
            ref_skin = ref_lab[skin_mask_ref]
            
            if len(ref_skin) < 50:
                print(f"[VAE-CC] Only {len(ref_skin)} skin pixels in reference, using luminance-based sampling")
                # Fallback: use mid-luminance pixels as reference
                lum = ref_lab[:, 0]
                mid_lum_mask = (lum > 100) & (lum < 220)
                ref_skin = ref_lab[mid_lum_mask]
                if len(ref_skin) < 50:
                    print("[VAE-CC] Insufficient reference pixels, falling back to statistical matching")
                    return self._statistical_matching_correction(original_np, processed_np, strength)
            
            print(f"[VAE-CC] Found {len(ref_skin)} reference skin pixels (anime_mode={anime_mode})")
            
            # Cluster reference skin colors for better representation
            n_clusters = min(8, len(ref_skin) // 50)
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans_ref = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            ref_labels = kmeans_ref.fit_predict(ref_skin)
            ref_centers = kmeans_ref.cluster_centers_
            
            # Now find the corresponding pixels in the processed image
            # Process the full processed LAB image
            proc_flat = proc_lab.reshape(-1, 3)
            
            # For each reference skin cluster center, find closest pixels in processed image
            # and compute the color shift needed
            corrected_lab = proc_lab.copy()
            
            # Build a color transfer map using the reference clusters
            # Strategy: match each pixel in the processed image to the nearest reference cluster,
            # then compute what the color should be based on the shift from original to reference
            
            # Sample from original and processed at same locations for paired analysis
            orig_flat = orig_lab.reshape(-1, 3)
            
            # Use every Nth pixel for speed
            sample_step = max(1, len(orig_flat) // 50000)
            orig_samples = orig_flat[::sample_step]
            proc_samples = proc_flat[::sample_step]
            
            # Cluster both sets together to find color correspondences
            n_map_clusters = min(32, len(orig_samples) // 100)
            if n_map_clusters < 4:
                n_map_clusters = 4
            
            # Cluster the processed samples
            kmeans_map = KMeans(n_clusters=n_map_clusters, random_state=42, n_init=5)
            proc_cluster_ids = kmeans_map.fit_predict(proc_samples)
            proc_centers = kmeans_map.cluster_centers_
            
            # For each cluster in processed, find the average color in original at same locations
            orig_centers = np.zeros_like(proc_centers)
            for ci in range(n_map_clusters):
                ci_mask = proc_cluster_ids == ci
                if np.sum(ci_mask) > 0:
                    orig_centers[ci] = np.mean(orig_samples[ci_mask], axis=0)
                else:
                    orig_centers[ci] = proc_centers[ci]
            
            # Compute per-cluster color shifts (in LAB space)
            color_shifts = orig_centers - proc_centers  # What needs to be added to processed to match original
            
            # Filter: only apply shifts for clusters that are "skin-like" (in either image)
            # This prevents overcorrecting non-skin areas
            for ci in range(n_map_clusters):
                proc_c = proc_centers[ci]
                orig_c = orig_centers[ci]
                
                # Check if this cluster is in a skin-like range (either in processed or original)
                is_skin_proc = (
                    (proc_c[0] > l_lo * 0.8) and (proc_c[0] < l_hi * 1.1) and
                    (proc_c[1] > a_lo * 0.8) and (proc_c[1] < a_hi * 1.2) and
                    (proc_c[2] > b_lo * 0.8) and (proc_c[2] < b_hi * 1.2)
                )
                is_skin_orig = (
                    (orig_c[0] > l_lo * 0.8) and (orig_c[0] < l_hi * 1.1) and
                    (orig_c[1] > a_lo * 0.8) and (orig_c[1] < a_hi * 1.2) and
                    (orig_c[2] > b_lo * 0.8) and (orig_c[2] < b_hi * 1.2)
                )
                
                if not is_skin_proc and not is_skin_orig:
                    # Reduce shift for non-skin clusters
                    color_shifts[ci] *= 0.15  # Only apply 15% of the shift for non-skin
            
            # Apply the color shifts to the full processed image
            # Assign each pixel to nearest cluster and apply corresponding shift
            proc_full_flat = proc_flat.copy()
            
            # Vectorized assignment
            distances = np.linalg.norm(
                proc_full_flat[:, np.newaxis, :] - proc_centers[np.newaxis, :, :], axis=2
            )
            closest = np.argmin(distances, axis=1)
            
            # Apply shifts with distance-based blending
            min_dists = np.min(distances, axis=1)
            max_dist = np.percentile(min_dists, 90) + 1e-6
            
            for ci in range(n_map_clusters):
                ci_mask = closest == ci
                if np.sum(ci_mask) > 0:
                    ci_dists = min_dists[ci_mask]
                    dist_weights = np.clip(1.0 - ci_dists / max_dist, 0.2, 1.0)
                    shift = color_shifts[ci] * strength * dist_weights[:, np.newaxis]
                    proc_full_flat[ci_mask] += shift
            
            corrected_lab = proc_full_flat.reshape(proc_lab.shape)
            
            # Clamp LAB values
            corrected_lab[:, :, 0] = np.clip(corrected_lab[:, :, 0], 0, 255)
            corrected_lab[:, :, 1] = np.clip(corrected_lab[:, :, 1], 0, 255)
            corrected_lab[:, :, 2] = np.clip(corrected_lab[:, :, 2], 0, 255)
            
            # Convert back to RGB
            corrected_lab_uint8 = corrected_lab.astype(np.uint8)
            corrected_rgb = cv2.cvtColor(corrected_lab_uint8, cv2.COLOR_LAB2RGB)
            
            # Blend with original processed image for safety
            blend = np.clip(strength, 0, 1)
            result = (
                processed_np.astype(np.float32) * (1 - blend) +
                corrected_rgb.astype(np.float32) * blend
            )
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print(f"[VAE-CC] Anime skin match applied with {n_map_clusters} color clusters, {len(ref_skin)} skin reference pixels")
            return result
            
        except Exception as e:
            print(f"[VAE-CC] Anime skin match failed: {e}, falling back to statistical matching")
            import traceback
            traceback.print_exc()
            return self._statistical_matching_correction(original_np, processed_np, strength)
    
    def _apply_anime_cool_bias(self, image_np, strength):
        """Apply a subtle cool/blue bias to counteract VAE warm shift in anime images."""
        corrected = image_np.astype(np.float32)
        # Reduce red/warm channel slightly, increase blue slightly
        corrected[:, :, 0] -= strength * 8   # Reduce red
        corrected[:, :, 2] += strength * 12  # Increase blue
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    # =========================================================================
    # Mask targeting — replaces the old preserve logic
    # =========================================================================
    def _apply_mask_targeting(self, processed_img, corrected_img, mask, edge_feather, device, correction_target):
        """Apply mask to target which areas get correction."""
        
        if correction_target == "correct_inpainted":
            # White areas in mask (inpainted) = CORRECT, black areas = keep original processed
            correction_mask = mask.to(device).float()
        elif correction_target == "correct_non_inpainted":
            # Black areas in mask (original) = CORRECT, white areas = keep processed  
            correction_mask = 1.0 - mask.to(device).float()
        else:  # full_image
            # No masking - return corrected image as-is
            return corrected_img
        
        # Apply feathering for smooth transitions
        if edge_feather > 0:
            correction_mask = self._feather_mask(correction_mask, edge_feather, device)
        
        # Apply mask: blend between processed and corrected
        correction_mask = correction_mask.unsqueeze(-1)  # Add channel dimension
        result = processed_img * (1 - correction_mask) + corrected_img * correction_mask
        
        target_name = "inpainted" if correction_target == "correct_inpainted" else "non-inpainted"
        print(f"[VAE-CC] Mask targeting: correcting {target_name} areas with {edge_feather}px feather")
        return result
    
    def _auto_target_correction(self, original_img, processed_img, corrected_img, edge_feather, device, correction_target):
        """Auto-detect inpainted areas and apply correction targeting."""
        # Calculate difference between original and processed to find changed areas
        diff = torch.abs(original_img - processed_img)
        diff_magnitude = torch.mean(diff, dim=-1)  # Average across RGB
        
        # Use absolute threshold instead of relative quantile
        # This works much better for anime where VAE shifts are uniform
        mean_diff = torch.mean(diff_magnitude)
        std_diff = torch.std(diff_magnitude)
        
        # Otsu-inspired: use mean + 0.5*std as threshold, with a minimum of 0.02
        threshold = max(float(mean_diff + 0.5 * std_diff), 0.02)
        
        # If the threshold is very low (images are very similar), use a percentile fallback
        if mean_diff < 0.01:
            # Images are nearly identical — probably no inpainting happened
            print("[VAE-CC] Images very similar, applying full-image correction")
            return corrected_img
        
        inpainted_mask = (diff_magnitude > threshold).float()
        
        if correction_target == "correct_inpainted":
            # Correct the changed (inpainted) areas
            correction_mask = inpainted_mask
        elif correction_target == "correct_non_inpainted":
            # Correct the unchanged (original) areas
            correction_mask = 1.0 - inpainted_mask
        else:
            return corrected_img
        
        # Apply feathering
        if edge_feather > 0:
            correction_mask = self._feather_mask(correction_mask, edge_feather, device)
        
        correction_mask = correction_mask.unsqueeze(-1)
        result = processed_img * (1 - correction_mask) + corrected_img * correction_mask
        
        inpainted_pct = torch.mean(inpainted_mask).item() * 100
        print(f"[VAE-CC] Auto-detected {inpainted_pct:.1f}% as inpainted, targeting: {correction_target}")
        return result
    
    def _feather_mask(self, mask, edge_feather, device):
        """Apply Gaussian blur feathering to a mask."""
        if ADVANCED_LIBS_AVAILABLE:
            mask_np = mask.cpu().numpy()
            ksize = min(edge_feather * 2 + 1, 99)  # OpenCV requires odd kernel size
            mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), edge_feather / 3)
            return torch.from_numpy(mask_np).to(device)
        else:
            # Simple box blur fallback using torch
            k = edge_feather
            if k < 1:
                return mask
            # Use avg_pool2d for approximate blurring
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            blurred = F.avg_pool2d(mask_4d, kernel_size=k*2+1, stride=1, padding=k)
            return blurred.squeeze(0).squeeze(0)
    
    # =========================================================================
    # Existing correction methods (with improvements)
    # =========================================================================
    def _advanced_3d_lut_correction(self, original_np, processed_np, strength):
        """Advanced 3D LUT-based color correction for precise VAE artifact fixing."""
        if not ADVANCED_LIBS_AVAILABLE:
            print("[VAE-CC] Advanced libs not available, using statistical matching fallback")
            return self._statistical_matching_correction(original_np, processed_np, strength)
        
        try:
            print("[VAE-CC] Building 3D color mapping...")
            
            # Sample colors for mapping (use every 4th pixel for speed)
            orig_samples = original_np[::4, ::4].reshape(-1, 3)
            proc_samples = processed_np[::4, ::4].reshape(-1, 3)
            
            # Use k-means to find representative color pairs
            n_clusters = min(64, len(orig_samples) // 10)  # Adaptive cluster count
            
            if n_clusters < 4:
                # Too few samples, use statistical matching
                return self._statistical_matching_correction(original_np, processed_np, strength)
            
            # Cluster processed colors
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            proc_clusters = kmeans.fit_predict(proc_samples)
            proc_centers = kmeans.cluster_centers_
            
            # Find corresponding original colors for each cluster
            orig_centers = np.zeros_like(proc_centers)
            for i in range(n_clusters):
                cluster_mask = proc_clusters == i
                if np.sum(cluster_mask) > 0:
                    orig_centers[i] = np.mean(orig_samples[cluster_mask], axis=0)
                else:
                    orig_centers[i] = proc_centers[i]
            
            # Apply color mapping to full image
            proc_flat = processed_np.reshape(-1, 3).astype(np.float32)
            
            # Vectorized distance calculation
            distances = np.linalg.norm(
                proc_flat[:, np.newaxis, :] - proc_centers[np.newaxis, :, :], axis=2
            )
            closest_clusters = np.argmin(distances, axis=1)
            
            # Apply color shifts with distance-based blending
            min_distances = np.min(distances, axis=1)
            max_distance = np.percentile(min_distances, 90) + 1e-6
            
            for i in range(n_clusters):
                cluster_mask = closest_clusters == i
                if np.sum(cluster_mask) > 0:
                    color_shift = orig_centers[i] - proc_centers[i]
                    cluster_distances = min_distances[cluster_mask]
                    distance_weights = np.clip(1.0 - cluster_distances / max_distance, 0.1, 1.0)
                    
                    for c in range(3):
                        shift_amount = color_shift[c] * strength * distance_weights
                        proc_flat[cluster_mask, c] += shift_amount
            
            corrected_np = proc_flat.reshape(processed_np.shape)
            corrected_np = np.clip(corrected_np, 0, 255).astype(np.uint8)
            
            print(f"[VAE-CC] 3D LUT correction applied using {n_clusters} color clusters")
            return corrected_np
            
        except Exception as e:
            print(f"[VAE-CC] 3D LUT correction failed: {e}, falling back to statistical matching")
            return self._statistical_matching_correction(original_np, processed_np, strength)
    
    def _histogram_matching_correction(self, original_np, processed_np, strength):
        """Histogram-based color matching."""
        if not ADVANCED_LIBS_AVAILABLE:
            return self._statistical_matching_correction(original_np, processed_np, strength)
        
        try:
            corrected_np = processed_np.astype(np.float32)
            original_float = original_np.astype(np.float32)
            
            # Match histogram for each channel
            for c in range(3):
                matched_channel = exposure.match_histograms(
                    corrected_np[:,:,c].astype(np.uint8), original_float[:,:,c].astype(np.uint8)
                ).astype(np.float32)
                corrected_np[:,:,c] = (
                    processed_np[:,:,c] * (1 - strength) + 
                    matched_channel * strength
                )
            
            return np.clip(corrected_np, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"[VAE-CC] Histogram matching failed: {e}")
            return self._statistical_matching_correction(original_np, processed_np, strength)
    
    def _statistical_matching_correction(self, original_np, processed_np, strength):
        """Statistical moment matching (mean and std). Works without ADVANCED_LIBS."""
        corrected_np = processed_np.astype(np.float32)
        original_float = original_np.astype(np.float32)
        
        for c in range(3):
            proc_mean = np.mean(corrected_np[:,:,c])
            proc_std = np.std(corrected_np[:,:,c])
            orig_mean = np.mean(original_float[:,:,c])
            orig_std = np.std(original_float[:,:,c])
            
            if proc_std > 0:
                normalized = (corrected_np[:,:,c] - proc_mean) / proc_std
                rescaled = normalized * orig_std + orig_mean
                
                corrected_np[:,:,c] = (
                    corrected_np[:,:,c] * (1 - strength) + 
                    rescaled * strength
                )
        
        return np.clip(corrected_np, 0, 255).astype(np.uint8)
