#!/usr/bin/env python3

# Libraries
import cv2
import os
import json
from typing import Dict, Any, Optional


class StereoCalibration:
    def __init__(self) -> None:
        self.__calibration_data = {
            "R": None,
            "T": None,
            "E": None,
            "F": None,
            "R1": None,
            "R2": None,
            "P1": None,
            "P2": None,
            "Q": None,
            "roi1": None,
            "roi2": None,
            "ret": None,
            "mtx_1": None,
            "dist_1": None,
            "mtx_2": None,
            "dist_2": None,
        }

    def calibrate(
        self,
        camera_1_parameters: Dict[str, Any],
        camera_2_parameters: Dict[str, Any],
        criteria=None,
        flags=None,
        alpha: Optional[int] = -1,
        newImageSize=None,
    ):

        if camera_1_parameters["image_size"] != camera_2_parameters["image_size"]:
            raise ValueError(
                f'camera_1 image size {camera_1_parameters["img_size"]} does not match camera_2 image size {camera_2_parameters["img_size"]}'
            )
        if criteria == None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)
        if flags == None:
            flags = cv2.CALIB_FIX_INTRINSIC
        objpoints = camera_1_parameters["obj_points"]
        imgpoints_1 = camera_1_parameters["img_points"]
        imgpoints_2 = camera_2_parameters["img_points"]
        mtx_1 = camera_1_parameters["mtx"]
        dist_1 = camera_1_parameters["dist"]
        mtx_2 = camera_2_parameters["mtx"]
        dist_2 = camera_2_parameters["dist"]

        # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
        ret, mtx_1, dist_1, mtx_2, dist_2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_1,
            imgpoints_2,
            mtx_1,
            dist_1,
            mtx_2,
            dist_2,
            camera_1_parameters["image_size"],
            criteria=criteria,
            flags=flags,
        )

        # Knowing the transformation between the two cameras (extrinsice parameters), we can perform stereo rectification
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            mtx_1,
            dist_1,
            mtx_2,
            dist_2,
            camera_1_parameters["image_size"],
            R,
            T,
            alpha=alpha,  # rectify_scale: if 0 image croped, if 1 image not croped
            newImageSize=newImageSize,
        )

        self.__calibration_data["R"] = R
        self.__calibration_data["T"] = T
        self.__calibration_data["E"] = E
        self.__calibration_data["F"] = F
        self.__calibration_data["R1"] = R1
        self.__calibration_data["R2"] = R2
        self.__calibration_data["P1"] = P1
        self.__calibration_data["P2"] = P2
        self.__calibration_data["Q"] = Q
        self.__calibration_data["roi1"] = validPixROI1
        self.__calibration_data["roi2"] = validPixROI1
        self.__calibration_data["ret"] = ret
        self.__calibration_data["mtx_1"] = mtx_1
        self.__calibration_data["dist_1"] = dist_1
        self.__calibration_data["mtx_2"] = mtx_2
        self.__calibration_data["dist_2"] = dist_2

        return self.__calibration_data
    
    @property
    def calibration_data(self):
        return self.__calibration_data

    def save_params(self, path: str, filename: str):

        if os.path.exists(path + filename):
            os.remove(path + filename)

        try:
            with open(path + filename, "w+") as outfile:

                __calibration_data = {
                    "R": [],
                    "T": [],
                    "R1": [],
                    "R2": [],
                    "P1": [],
                    "P2": [],
                    "Q": [],
                }

                __calibration_data["R"] = [
                    self.__calibration_data["R"].flatten()[i] for i in range(9)
                ]
                __calibration_data["T"] = [
                    self.__calibration_data["T"].flatten()[i] for i in range(3)
                ]

                __calibration_data["R1"] = [
                    self.__calibration_data["R1"].flatten()[i] for i in range(9)
                ]
                __calibration_data["R2"] = [
                    self.__calibration_data["R2"].flatten()[i] for i in range(9)
                ]
                __calibration_data["P1"] = [
                    self.__calibration_data["P1"].flatten()[i] for i in range(12)
                ]
                __calibration_data["P2"] = [
                    self.__calibration_data["P2"].flatten()[i] for i in range(12)
                ]

                __calibration_data["Q"] = [
                    self.__calibration_data["Q"].flatten()[i] for i in range(16)
                ]

                json.dump(__calibration_data, outfile)
        except FileNotFoundError:
            raise FileNotFoundError(f"The {path+filename} directory does not exist")
