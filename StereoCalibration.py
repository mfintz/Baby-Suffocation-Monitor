#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2

BOARD_H = 7
BOARD_W = 3

if __name__ == '__main__':
    debug_dir = "C:\TargetFolder" #TODO: Change this!

    square_size = 1.0
    frame_h, frame_w = 0, 0
    pattern_size = (BOARD_H, BOARD_W)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    obj1_points = []
    img1_points = []
    obj2_points = []
    img2_points = []

    i = 1

    # Take 30 pictures
    while i <= 30:
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()

        if frame1 is not None and frame2 is not None:
            cv2.imshow('Camera1', frame1)
            cv2.imshow('Camera2', frame2)

            frame_h, frame_w = frame1.shape[:2]

            if cv2.waitKey(1) & 0xFF == ord('c'):
                found, corners1 = cv2.findChessboardCorners(frame1, pattern_size)
                if not found:
                    print("No chessboard in camera 1. Please try again")
                    continue
                else:
                    # Find chessboard pattern
                    found, corners2 = cv2.findChessboardCorners(frame2, pattern_size)

                    if not found:
                        print("No chessboard in camera 2. Please try again")
                        continue
                    else:
                        # Save images
                        cv2.imwrite("cap1_" + str(i) + ".jpg", frame1)
                        cv2.imwrite("cap2_" + str(i) + ".jpg", frame2)

                        # Add corners to arrays
                        img1_points.append(corners1.reshape(-1, 2))
                        img2_points.append(corners2.reshape(-1, 2))
                        obj1_points.append(pattern_points)
                        obj2_points.append(pattern_points)

                        print("Captured frame " + str(i))
                        i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Perform calibration
    rms1, camera_matrix1, dist_coefs1, _, _ = cv2.calibrateCamera(obj1_points, img1_points, (frame_w, frame_h), None, None)
    rms2, camera_matrix2, dist_coefs2, _, _ = cv2.calibrateCamera(obj2_points, img2_points, (frame_w, frame_h), None, None)

    # Print results
    print("RMS 1:" + str(rms1))
    print("RMS 2:" + str(rms2))

    print("Camera matrix 1:")
    print(camera_matrix1)
    print("Camera matrix 2:")
    print(camera_matrix2)
