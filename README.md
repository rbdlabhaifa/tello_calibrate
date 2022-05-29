# Calibration of the camera of the Tello drone

This script runs in two steps:
- Capturing "chessboard pattern" images from Tello drone. Just run the script and it will capture and save the images. 
- Finding the camera matrix and distortion coefficients of the camera based on captured images.