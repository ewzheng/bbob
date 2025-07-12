import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import random

# -----------------------------------------------------------------------------
# Geometric augmentation – multi-scale random crop (Pix2Seq-style)
# -----------------------------------------------------------------------------

TARGET_SIZE = (256, 256)

# -------------------------------------------------------------
# Multi-scale crop that *guarantees* at least one bbox survives.
# -------------------------------------------------------------
# We first apply RandomResizedCrop (scale jitter + slight AR jitter) and then
# force the final window to stay near a randomly chosen GT box by means of
# RandomCropNearBBox.  This reproduces Pix2Seq-style MS crop but avoids the
# "empty GT" problem.

_ms_crop_aug = A.Compose(
    [
        # scale jitter + aspect-ratio jitter
        A.RandomResizedCrop(
            size=TARGET_SIZE,
            scale=(0.4, 1.0),
            ratio=(0.75, 1.33),
            p=1.0,
        ),
        # make sure at least one box remains visible
        A.RandomCropNearBBox(
            max_part_shift=0.2,
            p=1.0,
        ),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["class_labels"],
        min_visibility=0.2,
    ),
)

def apply_ms_crop(image, boxes, labels, *, scale_range=(0.4, 1.0)):
    """Apply Pix2Seq-style random scale jitter + crop to image and boxes.

    Parameters
    ----------
    image   : PIL.Image | np.ndarray | torch.Tensor(3,H,W)
    boxes   : list[list[float]]  xywh top-left normalised 0‥1
    labels  : list[str]
    scale_range : tuple(float,float)
        Passed to `RandomResizedCrop.scale`.
    Returns
    -------
    aug_img, aug_boxes, aug_labels (boxes stay xywh top-left 0‥1)
    """
    # Make sure boxes & labels are lists
    boxes = boxes or []
    labels = labels or []

    # If caller wants a custom scale range, rebuild the transform lazily.
    # We still include RandomCropNearBBox so that ≥1 GT survives.
    if scale_range != (0.4, 1.0):
        # Validate scale_range; clamp to (0,1] if necessary to satisfy Albumentations
        lo, hi = scale_range
        lo = max(0.0, min(1.0, lo))
        hi = max(lo, min(1.0, hi))
        scale_range_valid = (lo, hi)
        aug = A.Compose(
            [
                A.RandomResizedCrop(
                    size=TARGET_SIZE,
                    scale=scale_range_valid,
                    ratio=(0.75, 1.33),
                    p=1.0,
                ),
                A.RandomCropNearBBox(
                    max_part_shift=0.2,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.2),
        )
    else:
        aug = _ms_crop_aug

    # Ensure numpy uint8 image for Albumentations
    img_np, was_pil = _to_numpy(image)

    # ------------------------------------------------------------------
    # If there are *no* GT boxes we cannot run RandomCropNearBBox → fall
    # back to RandomResizedCrop only (retain scale jitter so behaviour is
    # still deterministic for empty-box images).
    # ------------------------------------------------------------------

    if boxes:
        # normal path – boxes present
        try:
            res = aug(image=img_np, bboxes=boxes, class_labels=labels)
        except Exception:
            # Any error → fallback to original sample
            return image, boxes, labels
    else:
        # Build a simple jitter-only transform on the fly
        jitter_only = A.RandomResizedCrop(
            size=TARGET_SIZE,
            scale=scale_range if scale_range != (0.4, 1.0) else (0.4, 1.0),
            ratio=(0.75, 1.33),
            p=1.0,
        )
        try:
            res = jitter_only(image=img_np)
            res["bboxes"] = []
            res["class_labels"] = []
        except Exception:
            return image, boxes, labels

    img_out = _restore_type(res["image"], was_pil)
    boxes_out = res["bboxes"]
    labels_out = res["class_labels"]

    return img_out, boxes_out, labels_out



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