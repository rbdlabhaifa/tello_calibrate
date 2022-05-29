# Calibration of the Tello drone camera

This script runs in two steps:
- Capturing "checkerboard pattern" images from the Tello drone camera. Just run the script and it will capture and save the images. 
- Finding the camera matrix and distortion coefficients based on captured images.

Parameters:
- DO_CAPTURE: perform capturing of the images. Drone initializes and after 5 seconds capturing starts. Press Esc to stop.
- DO_CALIBRATE: perform calibration of the captured images.
- CHECKERBOARD_SIZE: size of the checkerboard pattern.
- FPS: number of frames to capture per second, approximately.
- CALIBRATION_PATH: folder, where captured images are saved.
- OUTPUT_PATH: name of the file, where "camera matrix", "distortion parameters", "image size" are saved on separate lines.

