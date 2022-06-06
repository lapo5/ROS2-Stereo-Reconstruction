#!/usr/bin/env python3

# Libraries
import sys 

from cv_bridge import CvBridge


import os
from time import sleep
import threading




import cv2
import numpy as np
import glob
from typing import Tuple
from tqdm import tqdm

class camera_calibration():
    

    @staticmethod
    def calibrate(images_path: str, chessboard_size: Tuple[int], frame_size: Tuple[int], **kwargs):
        '''
        static method to calibrate a single camera.
        
        images_path: path to the images.
        chessboard_size: tuple with chess board size.
        frame_size: tuple with width and height.
        
        '''
        
        """ Optional:  criteria = None, display = False """
        
        display = kwargs.get('display', False)
        criteria = kwargs.get('criteria', (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        obj_points = [] # 3d point in real world space
        img_points = [] # 2d points in image plane.
        
        if images_path is None: #assumes same directory
            images = glob.glob("*.png")
        else:
            images = glob.glob(os.path.join(images_path,'*.png'))
        
        if images == []:
            raise ValueError('No images found in specified directory.')
        
        corners_found = 0

        for fname in tqdm(images):
            
            img = cv2.imread(fname)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray_img, (chessboard_size[0], chessboard_size[1]), None)

            # If found, add object points, image points (after refining them)
            if ret == True:

                obj_points.append(objp)

                corners = cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)
                img_points.append(corners)
                
                cv2.drawChessboardCorners(img, chessboard_size, corners, ret)

                # Draw and display the corners
                if display:
                    cv2.imshow('Image', img)
                    cv2.waitKey(1000)
                corners_found += 1


        cv2.destroyAllWindows()
        
        ############## CALIBRATION #######################################################
        print(f"{corners_found} / {len(images)} images calibrated.")

        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frame_size, None, None)
        height, width, channels = img.shape
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
        
        return {'img_points':img_points, 'obj_points':obj_points, 'mtx':newCameraMatrix, 'dist':dist, 'rvecs':rvecs, 'tvecs':tvecs}