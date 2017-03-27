#!/usr/bin/env python

from __future__ import print_function

from math import atan2, sqrt, degrees
import numpy as np
import cv2
from time import sleep
from statistics import median


CAM1_DEVICE_INDEX = 0
CAM2_DEVICE_INDEX = 1

# Chessboard dimensions
BOARD_H = 7
BOARD_W = 3

COLOR_THRESH = 20 # RGB factor for searching specific color
SIZE_FACTOR = 1.5 # size factor for searching specific color sticker
DIST_FACTOR = 1.8 # distance factor (relative to distance to other stickers)
UNDEFINED_COORD = -999 # when specific color not found - give it this fake coordinate
COLORS_HALF = 2 # number of "neighbors" that some sticker must have in its range

DANGER_LIMIT = 3
X_ANGLE_LIMIT = 40
Y_ANGLE_LIMIT = 40
Z_ANGLE_LIMIT = 40


# frequency parameters
INTER_FRAME_SLEEP = 0.2
INTER_BURST_SLEEP = 1.5
FRAMES_PER_BURST = 10

# stickers data
colors1 = []
colors2 = []
boundaries1 = []
boundaries2 = []
init_cents1 = []
init_cents2 = []
init_areas1 = []
init_areas2 = []

lastX = 0
lastY = 0


def onmouse(event, x, y, flags, param):
    '''
    Save the coordinates of last clicked location (left mouse button).
    :param event: Mouse event
    :param x: X-coordinate
    :param y: Y-coordinate
    '''
    global lastX, lastY

    # Catch only left clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        lastX = x
        lastY = y


def onmouseEmpty(event, x, y, flags, param):
    '''
    Disable mouse event.
    '''
    pass


def pin_init_colors(init_frame, window_name):
    '''
    Let user pin stickers in the frame
    :return: init_pin: list of x&y coordinates of stickers
    :return: pin_colors: list of stickers colors
    '''
    pin_colors = []
    init_pin = []

    while True:
        cv2.imshow(window_name, init_frame)

        key = cv2.waitKey(0)
        # Space is pressed
        if key & 0xFF == 32:
            # Extract color from last clicked location
            init_pin.append([lastX, lastY])
            pin_colors.append(init_frame[lastY, lastX])
            print("Color number", len(pin_colors), "confirmed")
        # Enter is pressed
        elif key & 0xFF == 13:
            break

    return init_pin, pin_colors


def calc_contour_centroid(contour):
    '''
    Calculate the centroid of a contour (using moments).
    :param contour: Contour
    :return: Centroid
    '''
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def capture_init_colors(frame, init_pin, color_bounds):
    '''
    Find all initial colors pinned by the user
    :param frame: Frame
    :param init_pin: Initial locations of colors the user has pinned
    :param color_bounds: Ranges of colors to search for
    :return: General success indicator, centroids of the colors, bounding rectangles of color blobs, areas of color blobs
    '''
    init_centroids = []
    init_bounding_rects = []
    init_areas = []

    # Loop over the boundaries
    for i in range(len(color_bounds)):
        # Create NumPy arrays from the boundaries
        (lower, upper) = color_bounds[i]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # Find the colors within the specified boundaries and apply the mask
        color_mask = cv2.inRange(frame, lower, upper)

        # Blur for better results
        blur_mask = cv2.GaussianBlur(color_mask, (9, 9), 0)

        # Dilate for better results
        kernel = np.ones((5, 5), np.uint8)
        dilation_mask = cv2.dilate(blur_mask, kernel, iterations=1)

        # Find contours of color blobs
        _, contours, _ = cv2.findContours(dilation_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # No contours - no blobs
        if len(contours) == 0:
            return False, None, None, None

        areas = {}

        # Check found contours
        for cont in contours:
            # The blob should contain the user's original pin click
            if cv2.pointPolygonTest(cont, (init_pin[i][0], init_pin[i][1]), False) >= 0:
                # Save the blob's area
                areas[cv2.contourArea(cont)] = cont

        # Extract the biggest blob
        biggest_blob_conts = [areas[k] for k in sorted(areas.keys(), reverse=True)[:1]]

        # Blobs found
        if len(biggest_blob_conts) != 0:
            cont = biggest_blob_conts[0]
            # Calculate the centroid of the blob
            cent_x, cent_y = calc_contour_centroid(cont)

            # Save the blob's centroid, area and bounding rectangle
            init_centroids.append(np.array([[float(cent_x), float(cent_y)]]))
            init_areas.append(cv2.contourArea(cont))
            (x, y, w, h) = cv2.boundingRect(cont)
            init_bounding_rects.append((x, y, w, h))

        # No blobs found - can't detect the current color
        else:
            init_centroids.append(np.array([[float(UNDEFINED_COORD), float(UNDEFINED_COORD)]]))
            init_areas.append(-1)
            init_bounding_rects.append(None)
            # print("Color " + str(i) + " wasn't found")

    return True, init_centroids, init_bounding_rects, init_areas


def capture_curr_colors(frame, init_areas, color_bounds):
    curr_cents = []
    curr_bounding_rects = []

    # Loop over the boundaries
    for i in range(len(color_bounds)):
        # Create NumPy arrays from the boundaries
        (lower, upper) = color_bounds[i]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # Find the colors within the specified boundaries and apply the mask
        color_mask = cv2.inRange(frame, lower, upper)

        blur_mask = cv2.GaussianBlur(color_mask, (9, 9), 0)

        kernel = np.ones((5, 5), np.uint8)
        dilation_mask = cv2.dilate(blur_mask, kernel, iterations=1)

        _, contours, _ = cv2.findContours(dilation_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            curr_cents.append(np.array([[float(UNDEFINED_COORD), float(UNDEFINED_COORD)]]))
            curr_bounding_rects.append(None)
            # print("color " + str(i + 1) + " not found!")
            continue

        area_diffs = {}

        for cont in contours:
            if cv2.contourArea(cont) < SIZE_FACTOR * init_areas[i]:
                area_diffs[abs(cv2.contourArea(cont) - init_areas[i])] = cont

        # TODO: Add epsilone to duplicates

        closest_size_blob_conts = [area_diffs[k] for k in sorted(area_diffs.keys())[:1]]

        if len(closest_size_blob_conts) != 0:
            cont = closest_size_blob_conts[0]

            cent_x, cent_y = calc_contour_centroid(cont)
            curr_cents.append(np.array([[float(cent_x), float(cent_y)]]))
            (x, y, w, h) = cv2.boundingRect(cont)
            curr_bounding_rects.append((x, y, w, h))
        else:
            curr_cents.append(np.array([[float(UNDEFINED_COORD), float(UNDEFINED_COORD)]]))
            curr_bounding_rects.append(None)
            # print("color " + str(i + 1) + " not found!")

    return True, curr_cents, curr_bounding_rects


def get_intersecting_cent_indices(cents1, cents2):
    inter_indices = []

    for i in range(len(cents1)):
        # if cents1[i][0][0] != float(UNDEFINED_COORD) and cents2[i][0][0] != float(UNDEFINED_COORD):
        if cents1[i] is not None and cents2[i] is not None:
            inter_indices.append(i)

    return inter_indices


def draw_color_outline(frame, status):
    outlined_frame = frame.copy()
    outline_color = None
    outline_text = None

    frame_h = frame.shape[0] #height of the frame
    frame_w = frame.shape[1] #width of the frame

# add color frame according to safety status of the baby
    if status == "safe":
        outline_color = (0, 255, 0)
        outline_text = "SAFE"
    elif status == "warning":
        outline_color = (40, 127, 255)
        outline_text = "WARNING"
    else:   # danger
        outline_color = (0, 0, 255)
        outline_text = "DANGER"

    cv2.rectangle(outlined_frame, (0, 0), (frame_w, frame_h), outline_color, thickness=10)
    cv2.putText(outlined_frame, outline_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, outline_color, thickness=3)

    return outlined_frame


# Capture frames on two cameras simultaneously.
# If both have a recognized chessboard - return both lists of corners.
def board_stereo_capture(cap1, cap2):
    while True:
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()

        if frame1 is None or frame2 is None:
            return False, None, None

        # show images from both cameras
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)

        if cv2.waitKey(1) & 0xFF == 13:
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Find corners of camera 1 frame
            ret, corners1 = cv2.findChessboardCorners(gray_frame1, (BOARD_H, BOARD_W), None)

            if not ret:
                print("No chessboard in Camera 1")
                return False, None, None

            # Find corners of camera 2 frame
            ret, corners2 = cv2.findChessboardCorners(gray_frame2, (BOARD_H, BOARD_W), None)

            if not ret:
                print("No chessboard in Camera 2")
                return False, None, None

            return True, corners1, corners2


def init_color_stereo_recog(frame1, frame2, stereo_colors, stereo_init_pin):
    areas1 = None
    areas2 = None

    # Find centroids of camera 1 frame
    ret, cam1_centroids, bounding_rects1, areas1 = capture_init_colors(frame1, stereo_init_pin[0], calc_color_boundaries(stereo_colors[0]))

    if not ret:
        print("No colors found in Camera 1")
        return False, None, None, None, None

    # Find centroids of camera 2 frame
    ret, cam2_centroids, bounding_rects2, areas2 = capture_init_colors(frame2, stereo_init_pin[1], calc_color_boundaries(stereo_colors[1]))

    if not ret:
        print("No colors found in Camera 2")
        return False, None, None, None, None

    return True, cam1_centroids, cam2_centroids, [bounding_rects1, bounding_rects2], [areas1, areas2]


# Capture frames on two cameras simultaneously.
# If both have a recognized chessboard - return both lists of corners.
def color_stereo_capture(cap1, cap2, stereo_init_areas, stereo_colors, baby_status):
    cam1_centroids = []
    cam2_centroids = []

    while True:
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()

        if frame1 is None or frame2 is None:
            return False, None, None, None

        frame1_display = draw_color_outline(frame1, baby_status)
        frame2_display = draw_color_outline(frame2, baby_status)

        cv2.imshow('Camera 1', frame1_display)
        cv2.imshow('Camera 2', frame2_display)

        if cv2.waitKey(1):
            # Find centroids of camera 1 frame
            ret, cam1_centroids, bounding_rects1 = capture_curr_colors(frame1, stereo_init_areas[0], calc_color_boundaries(stereo_colors[0]))

            if not ret:
                print("No colors found in Camera 1")
                return False, None, None, None

            # Find centroids of camera 2 frame
            ret, cam2_centroids, bounding_rects2 = capture_curr_colors(frame2, stereo_init_areas[1], calc_color_boundaries(stereo_colors[1]))

            if not ret:
                print("No colors found in Camera 2")
                return False, None, None, None

            return True, [frame1, frame2], cam1_centroids, cam2_centroids, [bounding_rects1, bounding_rects2]


def calc_color_boundaries(colors):
    '''
    Create a boundary of COLOR_THRESH size around the BGR value of each color in the list.
    :param colors: list of BGR colors
    :return: list of boundaries
    '''
    boundaries = []

    for col in colors:
        # Calculate lower bounds
        lower = []
        lower.append(col[0] - COLOR_THRESH if col[0] - COLOR_THRESH >= 0 else 0)
        lower.append(col[1] - COLOR_THRESH if col[1] - COLOR_THRESH >= 0 else 0)
        lower.append(col[2] - COLOR_THRESH if col[2] - COLOR_THRESH >= 0 else 0)

        # Calculate upper bounds
        upper = []
        upper.append(col[0] + COLOR_THRESH if col[0] + COLOR_THRESH <= 255 else 255)
        upper.append(col[1] + COLOR_THRESH if col[1] + COLOR_THRESH <= 255 else 255)
        upper.append(col[2] + COLOR_THRESH if col[2] + COLOR_THRESH <= 255 else 255)

        boundaries.append((lower, upper))

    return boundaries


# run heuristics on found centroiids and return only the centroids that passed the distanse test
def verify_dist_validity(init_dist_mat, curr_cents):
    valid_cents = []

    for i in range(len(curr_cents)):
        # if color not found - append "none" to array of valid centroids
        if curr_cents[i][0][0] == UNDEFINED_COORD and curr_cents[i][0][1] == UNDEFINED_COORD:
            valid_cents.append(None)
        else:
            valid_neighbours = 0
            # calculate the number of neighbours that are within distance range to current color    
            for j in range(len(curr_cents)):
                if curr_cents[j][0][0] == UNDEFINED_COORD and curr_cents[j][0][1] == UNDEFINED_COORD:
                    continue
                # if 0.7 * init_dist_mat[i][j] <= np.linalg.norm(np.array(curr_cents[i]) - np.array(curr_cents[j])) <= 1.3 * init_dist_mat[i][j]:
                elif (0.3 * init_dist_mat[i][j] <= np.linalg.norm(np.array(curr_cents[i]) - np.array(curr_cents[j]))) and (np.linalg.norm(np.array(curr_cents[i]) - np.array(curr_cents[j])) <= 1.8 * init_dist_mat[i][j]):
                    valid_neighbours += 1

            # if the color has at least minimal number of valid neighbours - appent its centroid to valid centroid array
            if valid_neighbours >= COLORS_HALF:
                valid_cents.append(curr_cents[i])
            else:
                # print("Failed distance check")
                valid_cents.append(None)
    return valid_cents


def calc_centroid(points):
    '''
    Calculate a centroid of a set of points, by all axis.
    :param points: List of points
    :return: centroid
    '''
    return np.mean(points, 0)


def translate_points(points, origin):
    '''
    Perform a translation to a set of points by given origin coordinates.
    :param points: List of points
    :param origin: Origin coordinates
    :return: List of translated points
    '''
    # Convert to Numpy array
    np_points = np.asarray(points)
    np_origin = np.asarray(origin)
    trans_points = []

    # Move each point by the given origin
    for point in np_points:
        trans_points.append(point - np_origin)

    return np.asarray(trans_points)


def flatten(mat):
    '''
    Flatten a 3D matrix to remove its redundant middle dimension
    :param mat: Matrix
    :return: Flattened matrix
    '''
    list_result = []

    for row in mat:
        list_result.append(row[0])

    return np.asarray(list_result)


def decompose_rot_mat(rot):
    '''
    Extract rotation angles in radians around each axis from a rotation matrix.
    :param rot: Rotation matrix
    :return: rotation angles
    '''
    rot_x = atan2(rot[2, 1], rot[2, 2])
    rot_y = atan2(-rot[2, 0], sqrt((rot[2, 1]**2) + (rot[2, 2]**2)))
    rot_z = atan2(rot[1, 0], rot[0, 0])

    return rot_x, rot_y, rot_z


def kabsch(P, Q, calcTranslation=False, centP=None, centQ=None):
    '''
    Perform Kabsch's algorithm and return optimal rotation and translation
    '''
    # Pt * Q
    cov_mat = np.dot(np.transpose(flatten(P)), flatten(Q))

    d, u, vt = cv2.SVDecomp(cov_mat)

    rot = np.dot(np.transpose(vt), np.transpose(u))
    v = np.transpose(vt)

    # Check and fix special reflection case
    if cv2.determinant(rot) < 0:
        # rot[:, 2] = np.multiply(rot[:, 2], -1)
        v[:, 2] = np.multiply(v[:, 2], -1)
        rot = np.dot(v, np.transpose(u))

    t = None

    if calcTranslation:
        t = np.dot(-rot, np.transpose(centP)) + np.transpose(centQ)
    
    # return rotation ans translation matrixes
    return rot, t


if __name__ == '__main__':
    obj1_points = []
    img1_points = []
    obj2_points = []
    img2_points = []

    # Load stored intrinsic calibration data for both cameras
    camera_matrix1 = None
    camera_matrix2 = None
    dist_coefs1 = None
    dist_coefs2 = None

    # TODO: load from file / hardcoded
    camera_matrix1 = np.array([[840.82404933, 0., 323.53853504],
                               [0., 841.42643292, 258.51749066],
                               [0., 0., 1.]])
    camera_matrix2 = np.array([[844.46870729, 0., 316.14999239],
                               [0., 846.73639705, 222.38602167],
                               [0., 0., 1.]])

    # Camera capture objects
    cap1 = cv2.VideoCapture(CAM1_DEVICE_INDEX)
    cap2 = cv2.VideoCapture(CAM2_DEVICE_INDEX)

    # Define axis - build a meshgrid of all 3D coordinates on a chessboard of given size
    objp1 = np.zeros((BOARD_H * BOARD_W, 1, 3), np.float32)
    objp1[:, :, :2] = np.mgrid[0:BOARD_H, 0:BOARD_W].T.reshape(-1, 1, 2)
    objp2 = np.zeros((BOARD_H * BOARD_W, 1, 3), np.float32)
    objp2[:, :, :2] = np.mgrid[0:BOARD_H, 0:BOARD_W].T.reshape(-1, 1, 2)

    # Extrinsic calibration
    rvec1 = rvec2 = tvec1 = tvec2 = None
    print("Please provide a stereo image with a chessboard for extrinsic calibration\n")

    while True:
        ret, corners1, corners2 = board_stereo_capture(cap1, cap2)

        if not ret:
            continue

        # Find the rotation and translation vectors of both cameras
        _, rvec1, tvec1 = cv2.solvePnP(objp1, corners1, camera_matrix1, dist_coefs1)
        _, rvec2, tvec2 = cv2.solvePnP(objp2, corners2, camera_matrix2, dist_coefs2)
        break

    rot_mat1, _ = cv2.Rodrigues(rvec1)
    rot_mat2, _ = cv2.Rodrigues(rvec2)

    rt_mat1 = np.hstack((rot_mat1, tvec1))
    rt_mat2 = np.hstack((rot_mat2, tvec2))

    proj_mat1 = np.dot(camera_matrix1, rt_mat1)
    proj_mat2 = np.dot(camera_matrix2, rt_mat2)

    # Init frame stream
    cent1 = cent2 = None
    cv2.destroyAllWindows()

    print("Please take an init frame\n")

    frame1 = None
    frame2 = None

    while True:
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()

        if frame1 is None or frame2 is None:
            continue

        cv2.imshow('Camera 1 Feed', frame1)
        cv2.imshow('Camera 2 Feed', frame2)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()

    # Collect colors for init frame. Keep requesting until user confirms selection
    confirm = False
    while not confirm:
        print("Please pin all the colors in camera 1 frame:")
        print("Click on the color with left mouse button and then confirm selection with space.")
        print("Press Enter to finish selection for camera 1 frame and continue.\n")
        cv2.namedWindow('Init Frame 1')
        cv2.setMouseCallback('Init Frame 1', onmouse)

        # Extract colors from frame 1 by pinning
        init_pin1, pin_colors1 = pin_init_colors(frame1, 'Init Frame 1')

        cv2.setMouseCallback('Init Frame 1', onmouseEmpty)
        cv2.destroyAllWindows()

        print("Please pin all the colors in camera 2 frame")
        print("Click on the color with left mouse button and then confirm selection with space.")
        print("Press Enter to finish selection for camera 2 frame and continue.\n")
        cv2.namedWindow('Init Frame 2')
        cv2.setMouseCallback('Init Frame 2', onmouse)

        # Extract colors from frame 2 by pinning
        init_pin2, pin_colors2 = pin_init_colors(frame2, 'Init Frame 2')

        cv2.setMouseCallback('Init Frame 2', onmouseEmpty)
        cv2.destroyAllWindows()

        # Set extracted colors as global
        colors1 = pin_colors1
        colors2 = pin_colors2

        # Calculate boundaries around colors
        boundaries1 = calc_color_boundaries(colors1)
        boundaries2 = calc_color_boundaries(colors2)

        # Recognize color blobs and find their centroids
        ret, cent1, cent2, bounding_rects, areas = init_color_stereo_recog(frame1, frame2, [pin_colors1, pin_colors2], [init_pin1, init_pin2])

        if not ret:
            print("Fatal Error!")
            exit()

        # Save blob areas as global
        init_areas1 = areas[0]
        init_areas2 = areas[1]

        frame1_copy = frame1.copy()
        frame2_copy = frame2.copy()

        # Display recognized color blobs with bounding rectangles and centroids
        for i in range(len(bounding_rects[0])):
            (x, y, w, h) = bounding_rects[0][i]
            cv2.rectangle(frame1_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame1_copy, (int(cent1[i][0][0]), int(cent1[i][0][1])), 4, (220, 220, 220), -1)

        for i in range(len(bounding_rects[1])):
            (x, y, w, h) = bounding_rects[1][i]
            cv2.rectangle(frame2_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame2_copy, (int(cent2[i][0][0]), int(cent2[i][0][1])), 4, (220, 220, 220), -1)

        # Show recognition result and ask for confirmation
        print("\nWere all selected colors were found correctly?")
        print("Confirm - Enter")
        print("Reset - Escape\n")

        while True:
            cv2.imshow('Camera 1 Feed', frame1_copy)
            cv2.imshow('Camera 2 Feed', frame2_copy)

            key = cv2.waitKey(1)
            if key & 0xFF == 13:
                confirm = True
                break
            elif key & 0xFF == 27:
                break

        cv2.destroyAllWindows()

    init_cents1 = cent1
    init_cents2 = cent2

    # Build the distance matrix
    dist_mat = []

    for i in range(len(cent1)):
        dist_mat.append([])

        for j in range(len(cent1)):
            dist_mat[i].append(np.linalg.norm(np.asarray(cent1[i]) - np.asarray(cent2[j])))

    cent1_np = np.array(cent1)
    cent2_np = np.array(cent2)

    # Perform triangulation for the initial frame
    init_points_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, cent1_np, cent2_np)
    init_points_3d = cv2.convertPointsFromHomogeneous(np.transpose(init_points_4d))
    init_origin = calc_centroid(init_points_3d)
    init_points_centered = translate_points(init_points_3d, init_origin)

    # print("Init (x,y,z):")
    # print(init_origin)
    # print("\n")

    danger_bursts_count = 0
    baby_status = "safe"

    # Curr frame live stream
    while True:
        deg_list = [[], [], []]

        for frame_index in range(FRAMES_PER_BURST):
            ret, frames, cent1, cent2, rects = color_stereo_capture(cap1, cap2, [init_areas1, init_areas2], [colors1, colors2], baby_status)

            if ret:
                cent1 = verify_dist_validity(dist_mat, cent1)
                cent2 = verify_dist_validity(dist_mat, cent2)

                inter_indices = get_intersecting_cent_indices(cent1, cent2)
                if len(inter_indices) < 3:     #TODO: Change later!
                    # print("Too few points identified")
                    continue

                inter_cent1 = [cent1[i] for i in inter_indices]
                inter_cent2 = [cent2[i] for i in inter_indices]

                inter_cent1_np = np.array(inter_cent1)
                inter_cent2_np = np.array(inter_cent2)

                '''# -------DRAW---------
                frame1_copy = frames[0].copy()
                frame2_copy = frames[1].copy()

                while True:
                    cv2.imshow('Draw 1', frame1_copy)
                    cv2.imshow('Draw 2', frame2_copy)

                    for i in inter_indices:
                        (x, y, w, h) = rects[0][i]
                        cv2.rectangle(frame1_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        cv2.circle(frame1_copy, (int(cent1[i][0][0]), int(cent1[i][0][1])), 4, (220, 220, 220), -1)

                    for i in inter_indices:
                        (x, y, w, h) = rects[1][i]
                        cv2.rectangle(frame2_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        cv2.circle(frame2_copy, (int(cent2[i][0][0]), int(cent2[i][0][1])), 4, (220, 220, 220), -1)

                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        init_cents1 = cent1
                        init_cents2 = cent2
                        break
                # ---------------------'''

                curr_points_4d = None
                curr_points_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, inter_cent1_np, inter_cent2_np)
                curr_points_3d = cv2.convertPointsFromHomogeneous(np.transpose(curr_points_4d))

                corresp_init_points_3d = np.array([init_points_3d[i] for i in inter_indices])
                corresp_init_points_centered = translate_points(corresp_init_points_3d, init_origin)
                # curr_origin = calc_centroid(curr_points_3d)

                # Perform SVD to find the transformation of found points
                cent_corresp_init = calc_centroid(corresp_init_points_3d)
                cent_current = calc_centroid(curr_points_3d)
                corresp_init_centered = translate_points(corresp_init_points_3d, cent_corresp_init)
                current_mean_centered = translate_points(curr_points_3d, cent_current)

                rot, t = kabsch(corresp_init_centered, current_mean_centered, True, cent_corresp_init, cent_current)

                # Transformation matrix
                rt = np.hstack((rot, t))

                # Find the transformed centroid of the current set
                transformed_cent = cv2.transform(np.array([init_origin]), rt)

                curr_points_centered = flatten(translate_points(curr_points_3d, transformed_cent))

                # print("Curr (x,y,z):")
                # print(transformed_cent)

                rot, _ = kabsch(corresp_init_points_centered, curr_points_centered)

                rot_x, rot_y, rot_z = decompose_rot_mat(rot)

                rot_x_deg = degrees(rot_x)
                rot_y_deg = degrees(rot_y)
                rot_z_deg = degrees(rot_z)

                deg_list[0].append(rot_x_deg)
                deg_list[1].append(rot_y_deg)
                deg_list[2].append(rot_z_deg)

            sleep(INTER_FRAME_SLEEP)

        if len(deg_list[0]) == 0:
            # print("ALERT! Baby not found!")
            # Increase danger counter
            danger_bursts_count += 1
            # print("Danger counter =", danger_bursts_count)

            if danger_bursts_count == DANGER_LIMIT:
                baby_status = "danger"
                # print("DANGER!")
        else:
            # The baby was found - no danger - reset counter
            danger_bursts_count = 0

            # Calculate the median rotation angles of the burst
            med_x_deg = median(deg_list[0])
            med_y_deg = median(deg_list[1])
            med_z_deg = median(deg_list[2])

            # If the baby was found in an angle above defined limit - change to warning status
            if med_x_deg > X_ANGLE_LIMIT or med_x_deg < -X_ANGLE_LIMIT or \
                med_y_deg > Y_ANGLE_LIMIT or med_y_deg < -Y_ANGLE_LIMIT or \
                med_z_deg > Z_ANGLE_LIMIT or med_z_deg < -Z_ANGLE_LIMIT:
                baby_status = "warning"
                # print("WARNING!")
            # The angle is ok - the baby is safe
            else:
                baby_status = "safe"
                # print("Back to safety!")

            # print("Pitch, Roll, Yaw:")
            # print("%.2f" % median(deg_list[0]))
            # print("%.2f" % median(deg_list[1]))
            # print("%.2f" % median(deg_list[2]))

        sleep(INTER_BURST_SLEEP)

    cv2.destroyAllWindows()
