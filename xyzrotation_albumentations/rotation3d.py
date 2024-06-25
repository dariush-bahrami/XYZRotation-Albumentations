import random
from typing import Tuple

import cv2 as cv
import numpy as np
from albumentations.core.transforms_interface import DualTransform

from .xyzrotation import (
    ImageSize,
    get_transform_matrix,
    transform_image,
    transform_points,
)


class Rotation3D(DualTransform):
    def __init__(
        self,
        x_rotation_range: Tuple[float, float],
        y_rotation_range: Tuple[float, float],
        z_rotation_range: Tuple[float, float],
        border_mode: int = cv.BORDER_CONSTANT,
        interpolation_mode: int = cv.INTER_CUBIC,
        border_value: Tuple[float, float, float] = (0, 0, 0),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.x_rotation_range = x_rotation_range
        self.y_rotation_range = y_rotation_range
        self.z_rotation_range = z_rotation_range
        self.border_mode = border_mode
        self.interpolation_mode = interpolation_mode
        self.border_value = border_value

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        transform_matrix, (new_h, new_w) = get_transform_matrix(
            (img_h, img_w),
            x_rotation=random.uniform(*self.x_rotation_range),
            y_rotation=random.uniform(*self.y_rotation_range),
            z_rotation=random.uniform(*self.z_rotation_range),
            x_translate=0,
            y_translate=0,
        )
        return {
            "transform_matrix": transform_matrix,
            "after_transform_image_size": ImageSize(new_h, new_w),
        }

    def apply(self, image, *, transform_matrix, after_transform_image_size, **kwargs):
        return transform_image(
            image,
            transform_matrix,
            after_transform_image_size,
            cv2_warp_perspective_kwargs=dict(
                borderMode=self.border_mode,
                flags=self.interpolation_mode,
                borderValue=self.border_value,
            ),
        )

    def apply_to_bbox(
        self,
        bbox,
        *,
        transform_matrix,
        after_transform_image_size,
        rows,
        cols,
        **kwargs,
    ):
        # Unnormalize bbox
        xmin, ymin, xmax, ymax = bbox
        xmin, xmax = xmin * cols, xmax * cols
        ymin, ymax = ymin * rows, ymax * rows

        # Transform bbox corners
        corners = np.array(
            [
                # [0, 0],
                [xmin, ymin],
                [xmin, ymax],
                [xmax, ymax],
                [xmax, ymin],
            ],
            dtype=np.float32,
        )
        corners = transform_points(corners, transform_matrix)

        # Extract new bbox and normalize using new image size
        new_h, new_w = after_transform_image_size
        xmin = float(np.min(corners[:, 0])) / new_w
        ymin = float(np.min(corners[:, 1])) / new_h
        xmax = float(np.max(corners[:, 0])) / new_w
        ymax = float(np.max(corners[:, 1])) / new_h

        # cap bbox values to be in [0, 1] range
        xmin = max(0, min(xmin, 1))
        ymin = max(0, min(ymin, 1))
        xmax = max(0, min(xmax, 1))
        ymax = max(0, min(ymax, 1))

        return (xmin, ymin, xmax, ymax)

    def apply_to_keypoint(
        self,
        keypoint,
        *,
        transform_matrix,
        after_transform_image_size,
        rows,
        cols,
        **kwargs,
    ):
        # print(keypoints)
        # print(args)
        # print(kwargs)
        x, y = keypoint
        x, y = x * cols, y * rows
        new_x, new_y = transform_points(
            np.array([[x, y]], dtype=np.float32), transform_matrix
        )[0]
        new_x = new_x / after_transform_image_size[1]
        new_y = new_y / after_transform_image_size[0]

        # cap keypoint values to be in [0, 1] range
        new_x = max(0, min(new_x, 1))
        new_y = max(0, min(new_y, 1))
        return new_x, new_y

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "x_rotation_range",
            "y_rotation_range",
            "z_rotation_range",
            "border_mode",
            "interpolation_mode",
            "border_value",
        )
