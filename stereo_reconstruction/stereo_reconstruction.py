#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Dict, Any, Optional


class StereoReconstruction:
    def __init__(self, file: str, reconstruction_parameters: Dict[str, Any]) -> None:
        # Camera parameters to undistort and rectify images
        cv_file = cv2.FileStorage()
        cv_file.open(file, cv2.FileStorage_READ)

        self.stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
        self.stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
        self.stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
        self.stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()

        self.Q = cv_file.getNode("q").mat()

        self.stereo = cv2.StereoBM_create()

        # Setting the updated parameters before computing disparity map
        self.stereo.setNumDisparities(reconstruction_parameters["numDisparities"])
        self.stereo.setBlockSize(reconstruction_parameters["blockSize"])
        self.stereo.setPreFilterType(reconstruction_parameters["preFilterType"])
        self.stereo.setPreFilterSize(reconstruction_parameters["preFilterSize"])
        self.stereo.setPreFilterCap(reconstruction_parameters["preFilterCap"])
        self.stereo.setTextureThreshold(reconstruction_parameters["textureThreshold"])
        self.stereo.setUniquenessRatio(reconstruction_parameters["uniquenessRatio"])
        self.stereo.setSpeckleRange(reconstruction_parameters["speckleRange"])
        self.stereo.setSpeckleWindowSize(reconstruction_parameters["speckleWindowSize"])
        self.stereo.setDisp12MaxDiff(reconstruction_parameters["disp12MaxDiff"])
        self.stereo.setMinDisparity(reconstruction_parameters["minDisparity"])

        cv_file.release()

    def disparity_from_stereovision(self, img_left, img_right):

        img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Undistort and rectify images
        img_left_nice = cv2.remap(
            img_left_gray,
            self.stereoMapL_x,
            self.stereoMapL_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )
        img_right_nice = cv2.remap(
            img_right_gray,
            self.stereoMapR_x,
            self.stereoMapR_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )

        # Compute disparity map
        disparity_map = self.stereo.compute(img_left_nice, img_right_nice)

        # Show disparity map before generating 3D cloud to verify that point cloud will be usable.
        return disparity_map

    def pcl_from_disparity(self, disparity_map, img_left, img_right):

        # Get new downsampled width and height
        h, w = img_right.shape[:2]

        # Convert disparity map to float32 and divide by 16 as show in the documentation
        disparity_map = np.float32(disparity_map / 16.0)

        # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(
            disparity_map, self.Q, handleMissingValues=True
        )
        # Get color of the reprojected points
        colors = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # Get rid of points with value 0 (no depth)
        mask_map = disparity_map > disparity_map.min()

        # Mask colors and points.
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]

        output_points = output_points

        return output_points, output_colors