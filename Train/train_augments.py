import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import random


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
            "heavy": {"prob": 0.7, "strength": 0.8}
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
                        snow_point_lower=0.1,
                        snow_point_upper=0.3 * strength,
                        p=1.0,
                    )
                ]),
                "sunflare": A.Compose([
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_lower=0,
                        angle_upper=1,
                        num_flare_circles_lower=6,
                        num_flare_circles_upper=10,
                        src_radius=160,
                        src_color=(255, 255, 255),
                        p=1.0,
                    )
                ]),
                "fog": A.Compose([
                    A.RandomFog(
                        fog_coef_lower=0.3 * strength,
                        fog_coef_upper=0.7 * strength,
                        alpha_coef=0.08,
                        p=1.0,
                    )
                ]),
                "shadow": A.Compose([
                    A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_lower=2,
                        num_shadows_upper=int(4 * strength) + 2,
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
                "motion_blur": A.Compose([A.MotionBlur(blur_limit=blur_limit, allow_shifted=True, p=1.0)]),
                "blur": A.Compose([A.Blur(blur_limit=int(blur_limit * 0.7), p=1.0)]),
                "defocus": A.Compose([A.Defocus(radius=(1, int(blur_limit * 0.8)), alias_blur=(0.1, 0.3), p=1.0)]),
                "optical_distortion": A.Compose([
                    A.OpticalDistortion(distort_limit=0.05 * strength, shift_limit=0.02 * strength, p=1.0)
                ]),
                "grid_distortion": A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.1 * strength, p=1.0)]),
                "gauss_noise": A.Compose([A.GaussNoise(var_limit=(2, int(8 * strength)), mean=0, per_channel=True, p=1.0)]),
                "iso_noise": A.Compose([
                    A.ISONoise(color_shift=(0.01, 0.03 * strength), intensity=(0.1, 0.3 * strength), p=1.0)
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