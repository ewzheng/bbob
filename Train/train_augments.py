import albumentations as A
import numpy as np
from PIL import Image
import random

# -----------------------------------------------------------------------------
# Geometric augmentation – multi-scale random crop (Pix2Seq-style)
# -----------------------------------------------------------------------------

TARGET_SIZE = (512, 512)
MAX_RETRIES = 3

# -------------------------------------------------------------
# Multi-scale crop that *guarantees* at least one bbox survives.
# -------------------------------------------------------------
# We now rely on `RandomSizedBBoxSafeCrop`, which internally performs a
# RandomResizedCrop-style area/aspect-ratio sampling **and** guarantees that
# all bounding boxes remain completely inside the cropped window. This removes
# the need for an explicit `RandomCropNearBBox` while still avoiding the
# "empty GT" problem.

_ms_crop_aug = A.Compose(
    [
        A.RandomSizedBBoxSafeCrop(
            height=TARGET_SIZE[0],
            width=TARGET_SIZE[1],
            erosion_rate=0.0,  # keep boxes tight
            p=1.0,
        ),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["class_labels"],
        min_visibility=0.1,
    ),
)

def apply_ms_crop(image, boxes, labels, *, scale_range=(0.4, 1.0)):
    """Apply Pix2Seq-style random scale jitter + crop to image and boxes.
    
    This version ensures at least one GT box survives the augmentation.

    Parameters
    ----------
    image   : PIL.Image | np.ndarray | torch.Tensor(3,H,W)
    boxes   : list[list[float]]  xywh top-left normalised 0‥1
    labels  : list[str]
    scale_range : tuple(float,float)
        Passed to `RandomResizedCrop.scale`.
    max_retries : int
        Maximum number of attempts to ensure at least one box survives
        
    Returns
    -------
    aug_img, aug_boxes, aug_labels (boxes stay xywh top-left 0‥1)
    """
    # Make sure boxes & labels are lists
    boxes = boxes or []
    labels = labels or []

    # Ensure numpy uint8 image for Albumentations
    img_np, was_pil = _to_numpy(image)

    # ------------------------------------------------------------------
    # If there are *no* GT boxes, just apply scale jitter
    # ------------------------------------------------------------------
    if not boxes:
        # Validate scale_range
        lo, hi = scale_range
        lo = max(0.0, min(1.0, lo))
        hi = max(lo, min(1.0, hi))
        scale_range_valid = (lo, hi)
        
        jitter_only = A.RandomResizedCrop(
            size=TARGET_SIZE,
            scale=scale_range_valid,
            ratio=(0.75, 1.33),
            p=1.0,
        )
        try:
            res = jitter_only(image=img_np)
            img_out = _restore_type(res["image"], was_pil)
            return img_out, [], []
        except Exception:
            return image, boxes, labels

    # ------------------------------------------------------------------
    # Strategy: Try multiple approaches to ensure at least one box survives
    # ------------------------------------------------------------------
    
    # Validate scale_range
    lo, hi = scale_range
    lo = max(0.0, min(1.0, lo))
    hi = max(lo, min(1.0, hi))
    scale_range_valid = (lo, hi)
    
    # Strategy 1: Try with adaptive min_visibility
    for min_vis in [0.1, 0.05, 0.01]:  # Progressively lower thresholds
        aug = A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(
                    height=TARGET_SIZE[0],
                    width=TARGET_SIZE[1],
                    erosion_rate=0.0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
                min_visibility=min_vis,
            ),
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                res = aug(image=img_np, bboxes=boxes, class_labels=labels)
                if res["bboxes"]:  # At least one box survived
                    img_out = _restore_type(res["image"], was_pil)
                    return img_out, res["bboxes"], res["class_labels"]
            except Exception:
                continue
    
    # Strategy 2: If Strategy 1 fails, try with looser crops (higher erosion)
    for erosion in [0.1, 0.2, 0.3]:  # Progressively looser crops
        aug_no_near = A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(
                    height=TARGET_SIZE[0],
                    width=TARGET_SIZE[1],
                    erosion_rate=erosion,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
                min_visibility=0.01,  # Very low threshold
            ),
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                res = aug_no_near(image=img_np, bboxes=boxes, class_labels=labels)
                if res["bboxes"]:  # At least one box survived
                    img_out = _restore_type(res["image"], was_pil)
                    return img_out, res["bboxes"], res["class_labels"]
            except Exception:
                continue
    
    # Strategy 3: Last resort - just resize without cropping
    try:
        resize_only = A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1], p=1.0)
        res = resize_only(image=img_np, bboxes=boxes, class_labels=labels)
        img_out = _restore_type(res["image"], was_pil)
        return img_out, res["bboxes"], res["class_labels"]
    except Exception:
        pass
    
    # Final fallback: return original (this should rarely happen)
    print("Warning: MS crop augmentation failed completely, returning original sample")
    return image, boxes, labels

# ---------------------------------------------------------------------------
# Simple horizontal flip augment (always apply, no retries).
# ---------------------------------------------------------------------------


_hflip_aug = A.Compose(
    [A.HorizontalFlip(p=1.0)],
    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.0),
)


def apply_hflip(image, boxes, labels):
    """Return a horizontally flipped copy of ``image`` and its boxes.

    Parameters
    ----------
    image : PIL.Image | np.ndarray | torch.Tensor(3,H,W)
    boxes : list[list[float]]
        Bounding boxes in *normalised* xywh (0‥1) format.
    labels : list[str]
        Corresponding label strings.

    Returns
    -------
    img_out, boxes_out, labels_out  –  boxes stay xywh 0‥1.
    """

    # Ensure list types to avoid Albumentations complaints
    boxes = boxes or []
    labels = labels or []

    img_np, was_pil = _to_numpy(image)
    h, w = img_np.shape[:2]

    try:
        res = _hflip_aug(image=img_np, bboxes=boxes, class_labels=labels)
        img_flipped = res["image"]
        boxes_flipped = res["bboxes"]
        labels_flipped = res["class_labels"]
    except Exception as e:
        # Fallback: manual flip; boxes remain unchanged (mirrored coord)
        print(f"Horizontal flip failed via Albumentations: {e}; falling back to numpy flip.")
        img_flipped = np.fliplr(img_np)
        boxes_flipped = []
        for x, y, bw, bh in boxes:
            new_x = w - x - bw  # mirror left coordinate
            boxes_flipped.append([new_x, y, bw, bh])
        labels_flipped = labels

    # Back to normalised
    boxes_out = [[bx / w, by / h, bw / w, bh / h] for bx, by, bw, bh in boxes_flipped]

    img_out = _restore_type(img_flipped, was_pil)

    return img_out, boxes_out, labels_flipped


def _to_numpy(img):
    """Convert PIL image to numpy array if necessary."""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB")), True
    elif isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            if img.dtype in [np.float32, np.float64] and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return img, False
    else:
        raise TypeError("`image` must be a PIL.Image or numpy.ndarray")


def _restore_type(img_np, was_pil):
    """Return image as PIL or numpy depending on `was_pil`."""
    if was_pil:
        return Image.fromarray(img_np)
    return img_np


# Pre-compiled transform pipelines for better performance
class Augments:
    def __init__(self):
        self._weather_transforms = {}
        self._camera_transforms = {}
        self._compile_transforms()
    
    def _compile_transforms(self):
        """Pre-compile all transform combinations for different intensities."""
        intensity_params = {
            "light": {"prob": 0.3, "strength": 0.3},
            "medium": {"prob": 0.5, "strength": 0.5}, 
            "heavy": {"prob": 0.7, "strength": 0.7}
        }
        
        for intensity, params in intensity_params.items():
            strength = params["strength"]
            
            # Weather transforms
            self._weather_transforms[intensity] = {
                "rain": A.Compose([
                    A.RandomRain(
                        brightness_coefficient=0.9,
                        drop_width=1,
                        blur_value=1,
                        p=1.0,
                    )
                ]),
                "snow": A.Compose([
                    A.RandomSnow(
                        brightness_coeff=2.5,
                        snow_point_range=(0.1 * strength, 0.3 * strength),
                        p=1.0,
                    )
                ]),
                "sunflare": A.Compose([
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_range=(0.0, 1.0),
                        num_flare_circles_range=(6, 10),
                        src_radius=160,
                        src_color=(255, 255, 255),
                        p=1.0,
                    )
                ]),
                "fog": A.Compose([
                    A.RandomFog(
                        fog_coef_range=(0.3 * strength, 0.7 * strength),
                        alpha_coef=0.08,
                        p=1.0,
                    )
                ]),
                "shadow": A.Compose([
                    A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_limit=(2, int(4 * strength) + 2),
                        shadow_dimension=int(6 * strength) + 4,
                        p=1.0,
                    )
                ])
            }
            
            # Camera transforms
            blur_limit = {
                "light": 3,
                "medium": 5,
                "heavy": 7
            }[intensity]
            
            self._camera_transforms[intensity] = {
                "motion_blur": A.Compose([
                    A.MotionBlur(blur_limit=(3, blur_limit), p=1.0)
                ]),
                "blur": A.Compose([
                    A.Blur(blur_limit=(3, blur_limit), p=1.0)
                ]),
                "defocus": A.Compose([
                    A.Defocus(radius=(1, max(1, int(blur_limit * 0.8))), alias_blur=(0.1, 0.3), p=1.0)
                ]),
                "optical_distortion": A.Compose([
                    A.OpticalDistortion(distort_limit=0.05 * strength, p=1.0)
                ]),
                "grid_distortion": A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.1 * strength, p=1.0)]),
                "gauss_noise": A.Compose([
                    A.GaussNoise(
                        std_range=(0.02 * strength, 0.08 * strength),
                        mean_range=(0.0, 0.0),
                        per_channel=True,
                        p=1.0,
                    )
                ]),
                "iso_noise": A.Compose([
                    A.ISONoise(
                        color_shift=(0.01 * strength, 0.03 * strength),
                        intensity=(0.1 * strength, 0.3 * strength),
                        p=1.0,
                    )
                ]),
                "chromatic_aberration": A.Compose([
                    A.ChromaticAberration(
                        primary_distortion_limit=0.02 * strength,
                        secondary_distortion_limit=0.01 * strength,
                        mode="green_purple",
                        p=1.0,
                    )
                ]),
                "brightness_contrast": A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.05 * strength, contrast_limit=0.05 * strength, p=1.0)
                ]),
                "hsv_shift": A.Compose([
                    A.HueSaturationValue(
                        hue_shift_limit=int(3 * strength),
                        sat_shift_limit=int(5 * strength),
                        val_shift_limit=0,
                        p=1.0,
                    )
                ])
            }


# Global instance to avoid repeated compilation
_augmenter = Augments()


def apply_weather_augmentations(
    image,
    *,
    intensity="medium",
    cloud_shadows=True,
    max_augmentations=2,
):
    """Optimized version with pre-compiled transforms and optional limiting."""
    
    img_np, was_pil = _to_numpy(image)
    
    if intensity not in _augmenter._weather_transforms:
        raise ValueError(f"intensity must be one of {list(_augmenter._weather_transforms.keys())}")
    
    transforms = _augmenter._weather_transforms[intensity]
    transform_names = ["rain", "snow", "sunflare", "fog"]
    
    if cloud_shadows:
        transform_names.append("shadow")
    
    # Optionally limit the number of augmentations
    if max_augmentations and max_augmentations < len(transform_names):
        transform_names = random.sample(transform_names, max_augmentations)
    
    augmented_images = []
    
    for name in transform_names:
        try:
            aug = transforms[name](image=img_np)["image"]
            augmented_images.append(_restore_type(aug, was_pil))
        except Exception as e:
            print(f"Warning: {name} augmentation failed ({e}) – skipping.")
    
    return augmented_images if augmented_images else [image]


def apply_camera_augmentations(
    image,
    *,
    intensity="medium",
    motion_blur=True,
    dirty_lens=True,
    max_augmentations=1,
):
    """Optimized version with pre-compiled transforms and optional limiting."""
    
    img_np, was_pil = _to_numpy(image)
    
    if intensity not in _augmenter._camera_transforms:
        raise ValueError(f"intensity must be one of {list(_augmenter._camera_transforms.keys())}")
    
    transforms = _augmenter._camera_transforms[intensity]
    transform_names = []
    
    if motion_blur:
        transform_names.extend(["motion_blur", "blur", "defocus"])
    
    if dirty_lens:
        transform_names.extend([
            "optical_distortion", "grid_distortion", "gauss_noise", 
            "iso_noise", "chromatic_aberration", "brightness_contrast", "hsv_shift"
        ])
    
    # Optionally limit the number of augmentations
    if max_augmentations and max_augmentations < len(transform_names):
        transform_names = random.sample(transform_names, max_augmentations)
    
    if not transform_names:
        return [image]
    
    augmented_images = []
    
    for name in transform_names:
        try:
            aug = transforms[name](image=img_np)["image"]
            augmented_images.append(_restore_type(aug, was_pil))
        except Exception as e:
            print(f"Warning: {name} augmentation failed ({e}) – skipping.")
    
    return augmented_images if augmented_images else [image]


def apply_batch_augmentations(
    images,
    *,
    weather_intensity="medium",
    camera_intensity="medium",
    weather_enabled=True,
    camera_enabled=True,
    max_augmentations_per_type=2,
):
    """Apply augmentations to a batch of images more efficiently."""
    
    all_augmented = []
    
    for image in images:
        img_np, was_pil = _to_numpy(image)
        augmented_variants = []
        
        if weather_enabled:
            weather_augs = apply_weather_augmentations(
                image, 
                intensity=weather_intensity,
                max_augmentations=max_augmentations_per_type
            )
            augmented_variants.extend(weather_augs)
        
        if camera_enabled:
            camera_augs = apply_camera_augmentations(
                image,
                intensity=camera_intensity,
                max_augmentations=max_augmentations_per_type
            )
            augmented_variants.extend(camera_augs)
        
        all_augmented.extend(augmented_variants)
    
    return all_augmented