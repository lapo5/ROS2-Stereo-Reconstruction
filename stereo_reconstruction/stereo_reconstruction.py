#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Dict, Any, Optional


class StereoReconstruction:
    def __init__(self, file: str) -> None:
        # Camera parameters to undistort and rectify images
        cv_file = cv2.FileStorage()
        cv_file.open(file, cv2.FileStorage_READ)

        self.stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
        self.stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
        self.stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
        self.stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

        self.Q = cv_file.getNode('q').mat()
        
        cv_file.release()
        
    
    def disparity_from_stereovision(self, imgL, imgR):
        
        # Undistort and rectify images
        imgR = cv2.remap(imgR, self.stereoMapR_x, self.stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        imgL = cv2.remap(imgL, self.stereoMapL_x, self.stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                        

        # Downsample each image 3 times (because they're too big)
        # imgL = self._downsample_image(imgL, 3)
        # imgR = self._downsample_image(imgR, 3)

        imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


        # For each pixel algorithm will find the best disparity from 0
        # Larger block size implies smoother, though less accurate disparity map

        # Set disparity parameters
        # Note: disparity range is tuned according to specific parameters obtained through trial and error. 
        block_size = 5
        min_disp = -1
        max_disp = 31
        num_disp = max_disp - min_disp # Needs to be divisible by 16

        # Create Block matching object. 
        stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
            numDisparities = num_disp,
            blockSize = block_size,
            uniquenessRatio = 5,
            speckleWindowSize = 5,
            speckleRange = 2,
            disp12MaxDiff = 2,
            P1 = 8 * 3 * block_size**2,#8*img_channels*block_size**2,
            P2 = 32 * 3 * block_size**2) #32*img_channels*block_size**2)


        #stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=win_size)

        # Compute disparity map
        disparity_map = stereo.compute(imgLgray, imgRgray)

        # Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
        return disparity_map
        
    # Downsamples image x number (reduce_factor) of times. 
    def _downsample_image(image, reduce_factor):
        for i in range(0,reduce_factor):
            #Check if image is color or grayscale
            if len(image.shape) > 2:
                row,col = image.shape[:2]
            else:
                row,col = image.shape

            image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
        return image
        
    def pcl_from_disparity(self, disparity_map, imgL, imgR):
        
        # Get new downsampled width and height 
        h,w = imgR.shape[:2]

        # Convert disparity map to float32 and divide by 16 as show in the documentation
        disparity_map = np.float32(np.divide(disparity_map, 16.0))

        # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(disparity_map, self.Q, handleMissingValues=False)
        # Get color of the reprojected points
        colors = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

        # Get rid of points with value 0 (no depth)
        mask_map = disparity_map > disparity_map.min()

        # Mask colors and points. 
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]
        
        return output_points, output_colors
    
