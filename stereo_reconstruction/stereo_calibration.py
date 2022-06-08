#!/usr/bin/env python3

# Libraries
import cv2
import os
import json
from typing import Dict, Any, Optional


class StereoCalibration:
    def __init__(self) -> None:
        self.calib_params = {
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
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            mtx_1,
            dist_1,
            mtx_2,
            dist_2,
            camera_1_parameters["image_size"],
            R,
            T,
            alpha=alpha,
            newImageSize=newImageSize,
        )

        self.calib_params["R"] = R
        self.calib_params["T"] = T
        self.calib_params["E"] = E
        self.calib_params["F"] = F
        self.calib_params["R1"] = R1
        self.calib_params["R2"] = R2
        self.calib_params["P1"] = P1
        self.calib_params["P2"] = P2
        self.calib_params["Q"] = Q
        self.calib_params["roi1"] = validPixROI1
        self.calib_params["roi2"] = validPixROI1
        self.calib_params["ret"] = ret
        self.calib_params["mtx_1"] = mtx_1
        self.calib_params["dist_1"] = dist_1
        self.calib_params["mtx_2"] = mtx_2
        self.calib_params["dist_2"] = dist_2

        return self.calib_params

    def save_params(self, path: str, filename: str):

        if os.path.exists(path + filename):
            os.remove(path + filename)

        try:
            with open(path + filename, "w+") as outfile:

                calib_params = {
                    "R": [],
                    "T": [],
                    "R1": [],
                    "R2": [],
                    "P1": [],
                    "P2": [],
                    "Q": [],
                }

                calib_params["R"] = [
                    self.calib_params["R"].flatten()[i] for i in range(9)
                ]
                calib_params["T"] = [
                    self.calib_params["T"].flatten()[i] for i in range(3)
                ]

                calib_params["R1"] = [
                    self.calib_params["R1"].flatten()[i] for i in range(9)
                ]
                calib_params["R2"] = [
                    self.calib_params["R2"].flatten()[i] for i in range(9)
                ]
                calib_params["P1"] = [
                    self.calib_params["P1"].flatten()[i] for i in range(12)
                ]
                calib_params["P2"] = [
                    self.calib_params["P2"].flatten()[i] for i in range(12)
                ]

                calib_params["Q"] = [
                    self.calib_params["Q"].flatten()[i] for i in range(16)
                ]

                json.dump(calib_params, outfile)
        except FileNotFoundError:
            raise FileNotFoundError(f"The {path+filename} directory does not exist")
