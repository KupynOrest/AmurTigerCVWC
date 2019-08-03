import albumentations as albu
import cv2


def get_transform(size, transform_type="weak", min_visibility=0):
    """Creates transformation for COCO dataset
    Args:
        size (int): image size to return
        transform_type (str):
            'weak': resizes and normalizes image and bbox
            'strong': performs different image effects, resizes and normalizes image and bbox
        min_visibility (int): minimum fraction of area for a bounding box
    Returns:
        albu.core.transforms_interface.BasicTransform: image and bbox transformation
    """
    bbox_params = {
        'format': 'coco',
        'min_visibility': min_visibility,
        'label_fields': ['category_id']
    }

    augs = {'strong': albu.Compose([
        albu.Resize(size, size),
        albu.HorizontalFlip(),
        # albu.VerticalFlip(p=0.1),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=.4,
                              border_mode=cv2.BORDER_CONSTANT),
        albu.OneOf([
            albu.Blur(),
            albu.MotionBlur(),
            albu.MedianBlur(),
            albu.GaussianBlur(),
        ], p=0.2),
        albu.OneOf([
            albu.GaussNoise(var_limit=(10, 35)),
            albu.IAAAdditiveGaussianNoise(),
            albu.JpegCompression(quality_lower=50),
        ], p=0.2),
        albu.OneOf([
            albu.RandomRain(),
            # albu.RandomSunFlare(),
            albu.RandomShadow()
        ], p=0.15),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(alpha=(0.1, 0.3)),
            albu.IAAEmboss(alpha=(0.1, 0.4)),
            albu.RandomGamma(),
            albu.RandomBrightnessContrast(),
        ], p=0.2),
        # albu.HueSaturationValue(p=0.25),
    ], bbox_params=bbox_params),
        'weak': albu.Compose([albu.Resize(size, size),
                              # albu.HorizontalFlip(),
                              ], bbox_params=bbox_params),
    }

    aug_fn = augs[transform_type]
    normalize = albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    pipeline = albu.Compose([aug_fn,
                             # ])
                             normalize])

    return pipeline
