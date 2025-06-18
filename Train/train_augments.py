import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


def _to_numpy(img):  # returns (ndarray, was_pil)
    """Convert PIL image to numpy array if necessary.

    Returns
    -------
    np.ndarray
        Image in H×W×C (RGB) format suitable for Albumentations.
    bool
        Whether the original image was a PIL Image (needed for type restoration).
    """
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB")), True
    elif isinstance(img, np.ndarray):
        # Ensure uint8 type for Albumentations
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


def apply_weather_augmentations(
    image,
    *,
    intensity: str = "medium",
    cloud_shadows: bool = True,
) -> list:
    """Generate one augmented copy *per* weather transform.

    Returns a list of images: `[rain_img, snow_img, sunflare_img, fog_img, shadow_img?, …]`.
    The list length depends on the enabled transforms and `cloud_shadows` flag.

    Parameters
    ----------
    image
        Input image as ``PIL.Image`` or ``numpy.ndarray`` *(H×W×C, uint8)*.
    intensity
        Weather effect intensity: ``'light'``, ``'medium'``, or ``'heavy'``. Controls
        probability and strength of effects. Defaults to ``'medium'``.
    cloud_shadows
        Whether to include cloud shadow effects. Defaults to ``True``.

    Returns
    -------
    List[Any]
        List of augmented images.

    Examples
    --------
    >>> # In a data pipeline
    >>> def augment_sample(sample):
    ...     sample['image'] = apply_weather_augmentations(sample['image'])
    ...     # All other keys (labels, metadata, etc.) remain unchanged
    ...     return sample
    """

    img_np, was_pil = _to_numpy(image)

    # --- Configure intensity-based parameters -----------------------------------------
    intensity_params = {
        "light": {"prob": 0.3, "strength": 0.3},
        "medium": {"prob": 0.5, "strength": 0.5}, 
        "heavy": {"prob": 0.7, "strength": 0.8}
    }
    
    if intensity not in intensity_params:
        raise ValueError(f"intensity must be one of {list(intensity_params.keys())}")
    
    params = intensity_params[intensity]
    prob = params["prob"]
    strength = params["strength"]

    # --- Build individual transforms ---------------------------------------------------
    base_transforms = [
        (
            "rain",
            A.RandomRain(
                brightness_coefficient=0.9,
                drop_width=1,
                blur_value=1,
                p=1.0,
            ),
        ),
        (
            "snow",
            A.RandomSnow(
                brightness_coeff=2.5,
                snow_point_range=(0.1, 0.3 * strength),
                p=1.0,
            ),
        ),
        (
            "sunflare",
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                num_flare_circles_range=(6, 10),
                src_radius=160,
                src_color=(255, 255, 255),
                p=1.0,
            ),
        ),
        (
            "fog",
            A.RandomFog(
                fog_coef_range=(0.3 * strength, 0.7 * strength),
                alpha_coef=0.08,
                p=1.0,
            ),
        ),
    ]

    if cloud_shadows:
        base_transforms.append(
            (
                "shadow",
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(2, int(4 * strength) + 2),
                    shadow_dimension=int(6 * strength) + 4,
                    p=1.0,
                ),
            )
        )

    augmented_images = []

    for name, t in base_transforms:
        try:
            aug = A.Compose([t])(image=img_np)["image"]
            augmented_images.append(_restore_type(aug, was_pil))
        except Exception as e:
            print(f"Warning: {name} augmentation failed ({e}) – skipping.")

    # If every augmentation failed, fall back to original image
    if not augmented_images:
        augmented_images.append(image)

    return augmented_images

def apply_camera_augmentations(
    image,
    *,
    intensity: str = "medium",
    motion_blur: bool = True,
    dirty_lens: bool = True,
) -> list:
    """Apply camera-related augmentations to simulate real-world camera conditions.

    This function applies motion blur and dirty/smudged lens effects to simulate
    common camera issues like camera shake, movement, and lens contamination.
    Perfect for making training data more robust to real-world conditions.

    Returns a list with one image *per* enabled camera transform.

    Parameters
    ----------
    image
        Input image as ``PIL.Image`` or ``numpy.ndarray`` *(H×W×C, uint8)*.
    intensity
        Effect intensity: ``'light'``, ``'medium'``, or ``'heavy'``. Controls
        probability and strength of effects. Defaults to ``'medium'``.
    motion_blur
        Whether to include motion blur effects. Defaults to ``True``.
    dirty_lens
        Whether to include dirty/smudged lens effects. Defaults to ``True``.

    Returns
    -------
    List[Any]
        List of augmented images.

    Examples
    --------
    >>> # Light camera effects
    >>> aug_img = apply_camera_augmentations(img, intensity='light')
    >>> 
    >>> # Only motion blur, no lens effects
    >>> aug_img = apply_camera_augmentations(img, dirty_lens=False)
    >>>
    >>> # In a data pipeline
    >>> def augment_sample(sample):
    ...     sample['image'] = apply_camera_augmentations(sample['image'])
    ...     return sample
    """

    img_np, was_pil = _to_numpy(image)

    # --- Configure intensity-based parameters -----------------------------------------
    intensity_params = {
        "light": {"prob": 0.3, "strength": 0.3, "blur_limit": 3},
        "medium": {"prob": 0.5, "strength": 0.5, "blur_limit": 5}, 
        "heavy": {"prob": 0.7, "strength": 0.8, "blur_limit": 7}
    }
    
    if intensity not in intensity_params:
        raise ValueError(f"intensity must be one of {list(intensity_params.keys())}")
    
    params = intensity_params[intensity]
    prob = params["prob"]
    strength = params["strength"]
    blur_limit = params["blur_limit"]

    # Build individual transforms lists -------------------------------------------------
    camera_transforms = []

    if motion_blur:
        camera_transforms.extend(
            [
                ("motion_blur", A.MotionBlur(blur_limit=blur_limit, allow_shifted=True, p=1.0)),
                ("blur", A.Blur(blur_limit=int(blur_limit * 0.7), p=1.0)),
                (
                    "defocus",
                    A.Defocus(radius=(1, int(blur_limit * 0.8)), alias_blur=(0.1, 0.3), p=1.0),
                ),
            ]
        )

    if dirty_lens:
        camera_transforms.extend(
            [
                (
                    "optical_distortion",
                    A.OpticalDistortion(
                        distort_limit=0.05 * strength, p=1.0
                    ),
                ),
                (
                    "grid_distortion",
                    A.GridDistortion(num_steps=5, distort_limit=0.1 * strength, p=1.0),
                ),
                (
                    "gauss_noise",
                    A.GaussNoise(std_range=(2, int(8 * strength)), mean_range=(0, 0), per_channel=True, p=1.0),
                ),
                (
                    "iso_noise",
                    A.ISONoise(
                        color_shift=(0.01, 0.03 * strength),
                        intensity=(0.1, 0.3 * strength),
                        p=1.0,
                    ),
                ),
                (
                    "chromatic_aberration",
                    A.ChromaticAberration(
                        primary_distortion_limit=0.02 * strength,
                        secondary_distortion_limit=0.01 * strength,
                        mode="green_purple",
                        p=1.0,
                    ),
                ),
                (
                    "brightness_contrast",
                    A.RandomBrightnessContrast(
                        brightness_limit=0.05 * strength, contrast_limit=0.05 * strength, p=1.0
                    ),
                ),
                (
                    "hsv_shift",
                    A.HueSaturationValue(
                        hue_shift_limit=int(3 * strength),
                        sat_shift_limit=int(5 * strength),
                        val_shift_limit=0,
                        p=1.0,
                    ),
                ),
            ]
        )

    augmented_images = []

    if not camera_transforms:
        return [image]

    for name, t in camera_transforms:
        try:
            aug = A.Compose([t])(image=img_np)["image"]
            augmented_images.append(_restore_type(aug, was_pil))
        except Exception as e:
            print(f"Warning: {name} augmentation failed ({e}) – skipping.")

    if not augmented_images:
        augmented_images.append(image)

    return augmented_images
    