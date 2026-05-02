"""Image analysis and content detection functions.

MODIFIED: v1.3-anime-fix — Improved anime/webtoon detection.
Key changes:
- Relaxed anime scene detection thresholds (edge_density < 0.25, saturation range)
- Added anime-aware face detection fallback using color-based skin region detection
- Expanded skin LAB ranges to include cool/pastel anime skin tones
- Better handling of stylized art classification
"""

import typing
import numpy as np
import torch
from .imports import ADVANCED_LIBS_AVAILABLE
from .device_utils import get_preferred_device

if ADVANCED_LIBS_AVAILABLE:
    import cv2
    from sklearn.cluster import KMeans
    from skimage import filters
    from scipy import ndimage


def analyze_image_content(
    image_np: np.ndarray, device: torch.device = None
) -> typing.Dict[str, typing.Any]:
    """Advanced image analysis using computer vision with GPU acceleration.
    
    v1.3-anime-fix: Improved anime/webtoon detection with wider thresholds
    and fallback skin region detection.
    """
    if not ADVANCED_LIBS_AVAILABLE:
        return {
            "faces": [],
            "dominant_colors": [],
            "scene_type": "unknown",
            "lighting": "auto",
        }

    analysis = {}

    if device is None:
        device = get_preferred_device()
    
    # Face detection — try standard Haar first, then anime fallback
    faces = []
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Standard Haar cascade (works for realistic faces)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        detected = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(detected) > 0:
            faces = detected.tolist()
    except Exception:
        pass
    
    # Anime fallback: if no faces found, try LBP cascade (better for anime)
    if not faces:
        try:
            # Try to load LBP cascade which works better for some anime styles
            lbp_path = cv2.data.haarcascades.replace('haarcascades', 'lbpcascades') + 'lbpcascade_anime.xml'
            import os
            if os.path.exists(lbp_path):
                anime_cascade = cv2.CascadeClassifier(lbp_path)
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                detected = anime_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
                )
                if len(detected) > 0:
                    faces = detected.tolist()
        except Exception:
            pass
    
    # Anime fallback #2: Detect skin-like regions as "pseudo-faces" for color correction
    if not faces:
        try:
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            
            # Detect skin-like regions using expanded anime ranges
            # This helps downstream skin correction even without face detection
            l_ch = lab[:, :, 0]
            a_ch = lab[:, :, 1]
            b_ch = lab[:, :, 2]
            
            # Expanded skin ranges for anime (including cool/pastel tones)
            skin_mask = (
                (l_ch > 80) & (l_ch < 255) &
                (a_ch > 108) & (a_ch < 165) &
                (b_ch > 108) & (b_ch < 180)
            )
            
            # If significant skin area found in top portion, create a pseudo-face bounding box
            h, w = image_np.shape[:2]
            top_skin = skin_mask[:h//3, :]
            skin_pixel_count = np.sum(top_skin)
            
            if skin_pixel_count > (h // 3 * w * 0.05):  # At least 5% of top region is skin
                # Find bounding box of skin in top region
                skin_ys, skin_xs = np.where(top_skin)
                if len(skin_ys) > 0:
                    x_min = max(0, int(np.min(skin_xs)) - 20)
                    y_min = max(0, int(np.min(skin_ys)) - 20)
                    x_max = min(w, int(np.max(skin_xs)) + 20)
                    y_max = min(h // 3, int(np.max(skin_ys)) + 20)
                    # Add as a pseudo-face for downstream processing
                    faces = [[x_min, y_min, x_max - x_min, y_max - y_min]]
                    print(f"[Analysis] Anime skin region detected as pseudo-face: {faces[0]}")
        except Exception:
            pass
    
    analysis["faces"] = faces

    # Color analysis
    try:
        pixels = image_np.reshape(-1, 3)
        sample_size = min(10000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=5, init="k-means++")
        kmeans.fit(sample_pixels)
        dominant_colors = kmeans.cluster_centers_
        analysis["dominant_colors"] = dominant_colors.tolist()
    except Exception:
        analysis["dominant_colors"] = []

    # Scene type classification — IMPROVED for anime/webtoon
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = filters.sobel(gray)
        edge_density = np.mean(edges)

        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        color_variance = np.std(hsv[:, :, 1])
        texture_contrast = np.std(gray)
        
        # v1.3-anime-fix: Relaxed thresholds for anime detection
        # Original: edge_density < 0.08 was too strict for line-art anime
        # Most webtoon/anime with ink outlines has edge_density 0.10-0.30
        if edge_density < 0.25 and avg_saturation > 60 and color_variance > 25:
            # Check if it's more like airbrushed anime vs line-art anime
            if edge_density < 0.12 and avg_saturation > 100:
                analysis["scene_type"] = "anime"  # Smooth/cel-shaded anime
            elif avg_saturation > 60 and texture_contrast < 60:
                analysis["scene_type"] = "anime"  # Line-art anime/webtoon
            elif color_variance > 30 and avg_saturation > 50:
                analysis["scene_type"] = "anime"  # Webtoon style
        elif edge_density < 0.15 and avg_saturation > 80 and texture_contrast < 35:
            analysis["scene_type"] = "stylized_art"
        elif avg_saturation > 100 and color_variance > 50 and texture_contrast > 40:
            analysis["scene_type"] = "concept_art"
        elif edge_density > 0.25 and texture_contrast > 65 and avg_saturation > 90:
            analysis["scene_type"] = "detailed_illustration"
        elif len(analysis["faces"]) > 0 and edge_density < 0.25:
            analysis["scene_type"] = "portrait"
        elif edge_density > 0.15 and avg_saturation < 80:
            analysis["scene_type"] = "realistic_photo"
        else:
            analysis["scene_type"] = "general"
    except Exception:
        analysis["scene_type"] = "general"

    # Lighting analysis
    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        brightness = np.mean(l_channel)
        contrast = np.std(l_channel)

        if brightness < 85:
            analysis["lighting"] = "low_light"
        elif brightness > 170:
            analysis["lighting"] = "bright"
        elif contrast < 20:
            analysis["lighting"] = "flat"
        else:
            analysis["lighting"] = "good"
    except Exception:
        analysis["lighting"] = "auto"

    return analysis
