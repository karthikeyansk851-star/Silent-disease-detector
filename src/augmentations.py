"""Augmentations that simulate real-world, non-clinical smartphone captures:
poor lighting, motion blur, and skin-tone variation.
"""
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

from src.config import IMG_SIZE


class RandomLowLight:
    """Randomly darkens or unevenly lights the image."""

    def __init__(self, prob=0.4, factor_range=(0.4, 1.0)):
        self.prob = prob
        self.factor_range = factor_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            factor = random.uniform(*self.factor_range)
            img = ImageEnhance.Brightness(img).enhance(factor)
        return img


class RandomMotionBlur:
    """Randomly applies blur to simulate a shaky handheld phone camera."""

    def __init__(self, prob=0.3, radius_range=(1, 3)):
        self.prob = prob
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomSkinToneShift:
    """Randomly shifts hue/saturation slightly to simulate varied skin tones
    and different camera white-balance settings."""

    def __init__(self, prob=0.4, hue_range=(-0.03, 0.03), sat_range=(0.8, 1.2)):
        self.prob = prob
        self.hue_range = hue_range
        self.sat_range = sat_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            sat_factor = random.uniform(*self.sat_range)
            img = ImageEnhance.Color(img).enhance(sat_factor)
        return img


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        RandomLowLight(),
        RandomMotionBlur(),
        RandomSkinToneShift(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
