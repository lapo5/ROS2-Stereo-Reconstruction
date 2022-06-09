#!/usr/bin/env python3

# Libraries

import os
import json
import cv2
import numpy as np
import glob
from typing import Any, List, Dict


class CameraCalibration:
    def __init__(self) -> None:
        self._calibration_data = {
            "image_size": None,
            "valid_images": None,
            "img_points": None,
            "obj_points": None,
            "mtx": None,
            "dist": None,
            "mean_error": None,
        }

    def calibrate(
        self,
        images_path: str,
        chessboard_size: List[int],
        image_size: List[int],
        minimum_valid_images: int,
        **kwargs,
    ) -> Dict[str, Any]:

        display: bool = kwargs.get("display", False)
        criteria: Any = kwargs.get(
            "criteria", (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : chessboard_size[0], 0 : chessboard_size[1]
        ].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        if images_path is None:  # assumes same directory
            images = glob.glob("*.png")
        else:
            images = glob.glob(os.path.join(images_path, "*.png"))

        if images == []:
            raise ValueError("No images found in specified directory.")

        valid_images = 0

        for fname in images:

            img = cv2.imread(fname)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray_img, (chessboard_size[0], chessboard_size[1]), None
            )

            # If found, add object points, image points (after refining them)
            if ret == True:

                obj_points.append(objp)

                corners = cv2.cornerSubPix(
                    gray_img, corners, (11, 11), (-1, -1), criteria
                )
                img_points.append(corners)

                cv2.drawChessboardCorners(img, chessboard_size, corners, ret)

                # Draw and display the corners
                if display:
                    cv2.imshow("Image", img)
                    cv2.waitKey(1000)
                valid_images += 1
                if valid_images >= minimum_valid_images:
                    break
        if valid_images < minimum_valid_images:
            return {}

        cv2.destroyAllWindows()

        ############## CALIBRATION ##############
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        # height, width, channels = img.shape
        # new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

        # Evaluate the mean error i.e. the calibration reprojection error
        tot_error = 0
        for i in range(len(obj_points)):

            img_points_2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(img_points[i], img_points_2, cv2.NORM_L2) / len(
                img_points_2
            )

            tot_error += error

        mean_error = tot_error / len(obj_points)

        self._calibration_data["image_size"] = image_size
        self._calibration_data["valid_images"] = valid_images
        self._calibration_data["img_points"] = img_points
        self._calibration_data["obj_points"] = obj_points
        self._calibration_data["mtx"] = mtx
        self._calibration_data["dist"] = dist
        self._calibration_data["mean_error"] = mean_error

        return self._calibration_data
    
    @property
    def calibration_data(self):
        return self._calibration_data

    def save_params(self, path: str, filename: str):

        if os.path.exists(path + filename):
            os.remove(path + filename)

        try:
            with open(path + filename, "w+") as outfile:
                calib_params = {"mtx": [], "dist": []}
                calib_params["mtx"] = [
                    self._calibration_data["mtx"].flatten()[i] for i in range(9)
                ]
                calib_params["dist"] = [
                    self._calibration_data["dist"].flatten()[i] for i in range(5)
                ]
                json.dump(calib_params, outfile)
        except FileNotFoundError:
            raise FileNotFoundError(f"The {path+filename} directory does not exist")
