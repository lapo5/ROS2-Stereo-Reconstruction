#!/usr/bin/env python3

# Libraries
import sys 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
from time import sleep
import threading
from ament_index_python.packages import get_package_share_directory

CALIB_FILE = "stereo_calib_params.json"



class CalibrationNode(Node):
    def __init__(self):
        super().__init__("calibration_node")
        self.get_logger().info("Calibration node is awake...")

        # Parameters declarations
        self.declare_parameter("number_of_images_to_calibrate", 10)
        self.number_of_images_to_calibrate = int(self.get_parameter("number_of_images_to_calibrate").value)

        self.declare_parameter("minimum_valid_images", 10)
        self.min_valid_images = int(self.get_parameter("minimum_valid_images").value)

        self.declare_parameter("board_dim", [6, 8])
        self.board_dim = self.get_parameter("board_dim").value

        self.declare_parameter("width_image", 1000)
        self.width_image = self.get_parameter("width_image").value

        self.declare_parameter("height_image", 1000)
        self.height_image = self.get_parameter("height_image").value

        self.declare_parameter("square_size", "28.0")
        self.square_size = float(self.get_parameter("square_size").value)

        self.declare_parameter("subscribers.camera_left", "/camera_left/raw_frame")
        self.camera_left_topic = self.get_parameter("subscribers.camera_left").value

        self.declare_parameter("subscribers.camera_right", "/camera_right/raw_frame")
        self.camera_right_topic = self.get_parameter("subscribers.camera_right").value

        self.declare_parameter("auto_capture.mode", "True")
        self.auto_capture = self.get_parameter("auto_capture.mode").value

        self.declare_parameter("single_cameras_calibrated", False)
        self.single_cameras_calibrated = self.get_parameter("single_cameras_calibrated").value

        self.declare_parameter("auto_capture.time_for_frame", True)
        self.time_for_frame = self.get_parameter("auto_capture.time_for_frame").value

        self.declare_parameter("calibration_path", "auto")
        self.calibration_path = self.get_parameter("calibration_path").value

        self.CRITERIA_CALIB_STEREO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        self.stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

        self.alpha_stereo_rectify = 0.0

        self.show_feedback = False

        package_share_directory = get_package_share_directory('stereo_reconstruction')

        if self.calibration_path != "auto":
            self.calib_path = self.calibration_path
        else:
            self.calib_path = package_share_directory + "/calibration/"
        
        # Class attributes
        self.bridge = CvBridge()
        self.current_frame_left = []
        self.current_frame_right = []
        self.calib_params = {"R": [], "T": [], "R1": [], "R2": [], "P1": [], "P2": [], "Q": []}

        self.calib_pics_left = []
        self.calib_pics_right = []

        self.stop_acquisition = False

        self.frame_left_sub = self.create_subscription(Image, self.camera_left_topic, self.callback_frame_left, 1)

        self.frame_right_sub = self.create_subscription(Image, self.camera_right_topic, self.callback_frame_right, 1)

        self.thread1 = threading.Thread(target=self.calib_process, daemon=True)
        self.thread1.start()


    def calib_process(self):

        self.get_logger().info("Init Calibration Node...")
        while len(self.current_frame_left) == 0 or len(self.current_frame_right) == 0 :
            self.get_logger().warn("Waiting for the first frame acquisition...")
            sleep(1)

        self.calibrate_stereo()


    def callback_frame_left(self, msg):
        self.current_frame_left = self.bridge.imgmsg_to_cv2(msg)

    def callback_frame_right(self, msg):
        self.current_frame_right = self.bridge.imgmsg_to_cv2(msg)
        

    def take_calib_pics(self):

        if not self.auto_capture:
            self.get_logger().info("\n======== KEYBOARD COMMANDS ========\n\nq - quit pictures acquisition\nc - capture actual frame\n\n")
        else:
            self.get_logger().info("\nAuto-Capture - Taking 1 Frame every {0} secs \n\n".format(self.time_for_frame))
            self.timer = self.create_timer(self.time_for_frame, self.update_frames) 

        while True:
            horizontal_concat = np.concatenate((self.current_frame_left, self.current_frame_right), axis=1)
            cv2.imshow("LiveCamera -- LEFT & RIGHT", horizontal_concat)
            key = cv2.waitKey(1)

            if key == ord("q"):
                self.get_logger().info("Calibration process has been stopped.")
                return False
            if key == ord("c"):
                self.update_frames()
            if self.stop_acquisition:
                return True


    def update_frames(self):

        if self.stop_acquisition:
            if self.auto_capture:
                self.timer.destroy()
        else:
            self.calib_pics_left.append(self.current_frame_left)
            self.calib_pics_right.append(self.current_frame_right)

            self.get_logger().info("Picture " + str(len(self.calib_pics_left)) + " out of " + str(self.number_of_images_to_calibrate))
            
            if len(self.calib_pics_left) == self.number_of_images_to_calibrate:
                self.get_logger().info(f"{self.number_of_images_to_calibrate} have been taken successfully.\n")
                cv2.destroyAllWindows()
                self.stop_acquisition = True
                return True


    def calibrate_stereo(self):

        is_done = self.take_calib_pics()

        if not is_done:
            self.get_logger().info("Calibration failed: pictures have not been captured correctly.")
            exit(0)

        self.get_logger().info("Calibration process started")

        objp = np.zeros((self.board_dim[0] * self.board_dim[1], 3), np.float32)
        objp[:, :2] = self.square_size * np.mgrid[0:self.board_dim[1], 0:self.board_dim[0]].T.reshape(-1, 2)

        objpoints = [] 
        
        #Pixel coordinates of checkerboards
        imgpoints_left = [] 
        imgpoints_right = []

        self.count_images = 0
        
        #coordinates of the checkerboard in checkerboard world space.
        objpoints = [] # 3d point in real world space
        
        for frame_left, frame_right in zip(self.calib_pics_left, self.calib_pics_right):

            if len(frame_left.shape) == 3:
                gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            else:
                gray_left = frame_left

            if len(frame_right.shape) == 3:
                gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            else:
                gray_right = frame_right

            c_ret1, corners1 = cv2.findChessboardCorners(gray_left, (self.board_dim[1], self.board_dim[0]), None)
            c_ret2, corners2 = cv2.findChessboardCorners(gray_right, (self.board_dim[1], self.board_dim[0]), None)
        
            if c_ret1 == True and c_ret2 == True:
                corners1 = cv2.cornerSubPix(gray_left, corners1, (11, 11), (-1, -1), self.CRITERIA_CALIB_STEREO)
                corners2 = cv2.cornerSubPix(gray_right, corners2, (11, 11), (-1, -1), self.CRITERIA_CALIB_STEREO)
        
                if self.show_feedback:
                    cv2.drawChessboardCorners(frame_left, (self.board_dim[1], self.board_dim[0]), corners1, c_ret1)
                    cv2.drawChessboardCorners(frame_right, (self.board_dim[1], self.board_dim[0]), corners2, c_ret2)

                    horizontal_concat = np.concatenate((frame_left, frame_right), axis=1)
                    cv2.imshow('Calibration', horizontal_concat)
        
                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)
                self.count_images = self.count_images + 1
        
        self.get_logger().info("Number of Valid Images: {0}".format(self.count_images))

        self.get_logger().info("Obtaining Calibration Parameters")

        if self.count_images > self.min_valid_images:
            if self.single_cameras_calibrated:
                self.get_logger().info("Using Pre-Calibrated Single Cameras Params")
                package_share_directory = get_package_share_directory("stereo_reconstruction")
                self.calibration_camera_left_path = package_share_directory + "/calibration/calib_params_cam_left.json"
                self.calibration_camera_right_path = package_share_directory + "/calibration/calib_params_cam_right.json"

                with open(self.calibration_camera_left_path, "r") as readfile:
                    self.cam_left_params = json.load(readfile)

                with open(self.calibration_camera_right_path, "r") as readfile:
                    self.cam_right_params = json.load(readfile)

            else:
                self.get_logger().info("Re-calibrate Single Cameras Params")
                self.cam_left_params = self.calibrate_cam(objpoints, imgpoints_left, gray_left.shape[::-1], "calib_params_cam_left.json", "Left")
                self.cam_right_params = self.calibrate_cam(objpoints, imgpoints_right, gray_left.shape[::-1], "calib_params_cam_right.json", "Right")

            
            self.K_left  = np.array(self.cam_left_params["mtx"], dtype=float).reshape(3, 3)
            self.D_left  = np.array(self.cam_left_params["dist"], dtype=float)

            self.K_right = np.array(self.cam_right_params["mtx"], dtype=float).reshape(3, 3)
            self.D_right = np.array(self.cam_right_params["dist"], dtype=float)
            
            ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right, 
                self.K_left, self.D_left,
                self.K_right,   self.D_right, (self.width_image, self.height_image), criteria = self.CRITERIA_CALIB_STEREO, 
                flags = self.stereocalibration_flags)

            R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
                self.K_left, self.D_left, self.K_right, self.D_right, (self.width_image, self.height_image),
                 R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=self.alpha_stereo_rectify)

            self.get_logger().info("stereoCalibrate ret: {0}".format(ret))

            self.get_logger().info("\nstereoRectify roi_left:\n {0}".format(roi_left))
            self.get_logger().info("\nstereoRectify roi_right:\n {0}".format(roi_right))

            self.calib_params["R"] = [R.flatten()[i] for i in range(9)]
            self.calib_params["T"] = [T.flatten()[i] for i in range(3)]

            self.calib_params["R1"] = [R1.flatten()[i] for i in range(9)]
            self.calib_params["R2"] = [R2.flatten()[i] for i in range(9)]
            self.calib_params["P1"] = [P1.flatten()[i] for i in range(12)]
            self.calib_params["P2"] = [P2.flatten()[i] for i in range(12)]

            self.calib_params["Q"] = [Q.flatten()[i] for i in range(16)]

            if os.path.exists(self.calib_path+CALIB_FILE):
                self.get_logger().info(CALIB_FILE + " already exists. Overwrite")
                os.remove(self.calib_path+CALIB_FILE)

            with open(self.calib_path+CALIB_FILE, "w+") as outfile:
                json.dump(self.calib_params, outfile)

            self.get_logger().info("Calibration has been completed successfully.\nCalibration path: " + self.calib_path + CALIB_FILE)

        else:
            self.get_logger().info("Calibration has failed. Not enough valid images (minimum is {0}).".format(self.min_valid_images))


    def calibrate_cam(self, objpoints, imgpoints, img_shape, calib_file, camera_name):

        calib_params = {"mtx": [], "dist": []}

        # Obtain calibration parameters
        ret, calib_params["mtx"], calib_params["dist"], rvecs, tvecs = cv2.calibrateCamera(objpoints, \
            imgpoints, img_shape, None, None)

        # Evaluate the mean error i.e. the calibration reprojection error
        tot_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], calib_params["mtx"], calib_params["dist"])
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error

        self.get_logger().info("mean error: " + str(tot_error/len(objpoints)) + "pixels" + "\n")

        self.get_logger().info("Calibration of Camera {0} has been completed successfully.".format(camera_name))

        calib_params["mtx"] = [calib_params["mtx"].flatten()[i] for i in range(9)]
        calib_params["dist"] = [calib_params["dist"].flatten()[i] for i in range(5)]

        if os.path.exists(self.calib_path+calib_file):
            self.get_logger().info(calib_file + " already exists. Overwrite")
            os.remove(self.calib_path+calib_file)

        with open(self.calib_path+calib_file, "w+") as outfile:
            json.dump(calib_params, outfile)

        return calib_params




# Main loop function
def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Calibration Node stopped cleanly')
    except BaseException:
        node.get_logger().info('Exception in Calibration Node:', file=sys.stderr)
        raise
    finally:
        rclpy.shutdown() 


# Main
if __name__ == '__main__':
    main()
