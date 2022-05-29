import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from os import path, makedirs
from time import sleep, time
from re import split
from djitellopy import Tello


def capture_images_tello(calibration_path, fps):
    ''' 
        Captures "checkerboard pattern" images and stores them at the
        calibration_path. Tries to maintain given fps.
    '''
    images_path = path.join(calibration_path, "{:04d}.png")
    tello = Tello()
    tello.connect()
    tello.send_command_with_return("streamon")
    print("turning on stream and waiting 5 seconds...")
    sleep(5)
    frame = None
    frame_idx = 0
    frame_delay_ms = 1000 // fps
    while True:
        start_time = time()
        frame = tello.get_frame_read().frame
        if frame is None:
            continue
        cv2.imshow('Capturing', frame)
        if frame_idx == 0:
            print("Frame size: {}".format(frame.shape))
        if frame_idx % fps == 0:
            print("Capturing frame:", frame_idx)
        filename = images_path.format(frame_idx)
        cv2.imwrite(filename, frame)
        frame_idx += 1
        elapsed_time = time() - start_time
        remaining_time = int(max(1, frame_delay_ms - elapsed_time))
        k = cv2.waitKey(remaining_time)
        if k==27:    # Esc key to stop
            break


def calibrate_on_images(calibration_path, output_path, checkerboard_size):
    '''
        Finds calibration matrix of the camera.
        Based on: https://learnopencv.com/camera-calibration-using-opencv/
    '''
    file_list = glob(path.join(calibration_path, "*.png"))
    # Defining the dimensions of checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), 
                    np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard_size[0],
                            0:checkerboard_size[1]].T.reshape(-1, 2)
    # calibration loop
    for frame_idx, filename in enumerate(tqdm(file_list)):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard_size, (
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE))
        if ret:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                        criteria)            
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, checkerboard_size, corners2,
                                            ret)            
        cv2.imshow('Calibration images', img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    h,w = img.shape[:2]
    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix : \n", mtx)
    print("Distortion coefficients : \n", dist)
    print("Frame size : \n", gray.shape)
    with open(output_path, "w") as f:
        f.writelines(map(lambda x: str(np.asarray(x)) + '\n', 
                         (mtx, dist, gray.shape)))

def load_calibration(filepath):
    '''
        Example function that parses the resulting calibration file.
        filepath - path to the file with the calibration information.
        Returns: "camera matrix", "distortion coefficients", "image size"
    '''
    def line_to_floats(line):
        ''' Removes delimiters and splits the string '''
        return list(map(float, filter(None, split("\[+|\]+|\s", line))))
    
    K, distortion, dims = [], None, None
    with open(filepath, "r") as f:
        for idx, line in enumerate(f):
            if idx < 3: # calibration matrix K
                K.append(line_to_floats(line))
            elif idx == 4: # distiortion parameters
                distortion = line_to_floats(line)
            elif idx == 5: # dimensions of the image
                dims =  line_to_floats(line)
            else:
                break
    return K, distortion, dims


if __name__ == "__main__":
    DO_CAPTURE = True
    DO_CALIBRATE = True
    CHECKERBOARD_SIZE = (6,9)
    FPS = 5
    CALIBRATION_PATH = "out/calibration_img/"
    OUTPUT_PATH = "out/K.txt"
    if not path.exists(CALIBRATION_PATH):
        makedirs(CALIBRATION_PATH)
    if DO_CAPTURE:
        print("Capturing...")
        capture_images_tello(CALIBRATION_PATH, FPS)
    if DO_CALIBRATE:
        print("Calibrating...")
        calibrate_on_images(CALIBRATION_PATH, OUTPUT_PATH, CHECKERBOARD_SIZE)
    print("Done.")
