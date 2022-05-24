import cv2
import mediapipe as mp
import numpy as np
from math import sin, cos, atan2, pi, copysign
from misc import *

mpdraw = mp.solutions.drawing_utils
mpholistic = mp.solutions.holistic
holistic = mpholistic.Holistic()

cap = cv2.VideoCapture(0)


def rotate_landmark(center, pos, angle) -> tuple:
    x = pos.x - center.x
    y = pos.y - center.y
    s = sin(angle)
    c = cos(angle)
    return (center.x + x * c - y * s, center.y + x * s + y * c)


def float_point(landmark):
    return (landmark.x * WIDTH, landmark.y * HEIGHT)


def draw_eyes(left_eye_center, right_eye_center):
    r = 20
    width = 2
    cv2.circle(img, left_eye_center, r, (255, 255, 255), -1)
    cv2.circle(img, right_eye_center, r, (255, 255, 255), -1)
    cv2.circle(img, left_eye_center, r, (0, 0, 0), width)
    cv2.circle(img, right_eye_center, r, (0, 0, 0), width)

def draw_eye(center):
    r = 20
    width = 2
    cv2.circle(img, center, r, (255, 255, 255), -1)
    cv2.circle(img, center, r, (0, 0, 0), width)

def draw_closed_eye(p, angle, tilt):
    p1 = move(p, 30, angle-tilt)
    p2 = move(p, 30, -angle-tilt)
    points = np.array([p1, p, p2], dtype=np.int32)
    cv2.polylines(img, [points], False, (0, 0, 0), 3)



def draw_mouth(center, width, height, intersection, tilt):
    tilt *= 180/pi
    # bottom right
    cv2.ellipse(img, center, (width - intersection[0], height - intersection[1]), tilt, 0, 90, (0, 0, 0), 2)
    # bottom left
    cv2.ellipse(img, center, (intersection[0], height - intersection[1]), tilt, 90, 180, (0, 0, 0), 2)
    # top right
    cv2.ellipse(img, center, (width - intersection[0], intersection[1]), tilt, 0, -90, (0, 0, 0), 2)
    # top left
    cv2.ellipse(img, center, intersection, tilt, -90, -180, (0, 0, 0), 2)
    #cv2.ellipse(img, center, (20, 5), tilt * 180 / pi, 0, 360, (0, 0, 0), -1)

def draw_closed_mouth(center, width, height, lip_l, tilt, direction):
    tilt *= 180 / pi
    # right
    cv2.ellipse(img, center, (width - lip_l, height), tilt, 0, -90*direction, (0, 0, 0), 2)
    # left
    cv2.ellipse(img, center, (lip_l, height), tilt, -90*direction, -180*direction, (0, 0, 0), 2)

def draw(face, pose):
    left_eye_outer = point(face[33])
    right_eye_outer = point(face[263])

    face_center_line = point(face[168])

    shoulder_right = point(pose[11])
    shoulder_left = point(pose[12])

    #lip_u_o = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    #lip_l_o = [375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    #lip_u_i = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    #lip_l_i = [324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

    # mouth
    lip_u_center = point(face[13])
    lip_l_center = point(face[14])

    lip_u_left = point(face[81])
    lip_u_right = point(face[311])

    lip_l_left = point(face[178])
    lip_l_right = point(face[402])

    lip_left = point(face[78])
    lip_right = point(face[308])

    eye_dist = distance(left_eye_outer, right_eye_outer)

    m_points = np.array([lip_left, lip_l_left, lip_l_right, lip_right, lip_u_right, lip_u_left, lip_left], dtype=np.int32)
    cv2.polylines(img, [m_points], False, (0, 0, 0), 5)

    #nose_left = point(face[102])
    #nose_right = point(face[331])
    nose_top = point(face[1])
    #nose_bottom = point(face[2])

    #cv2.circle(img, lip_left, 2, (255, 0, 0), -1)

    cv2.circle(img, shoulder_left, 2, (255, 0, 0), -1)
    cv2.circle(img, shoulder_right, 2, (255, 0, 255), -1)

    tilt_b = atan2(shoulder_left[1]-shoulder_right[1], shoulder_left[0]-shoulder_right[0])

    center = (WIDTH//2, HEIGHT//2)

    #tilt_h = min(max(atan2(right_eye_outer[1]-left_eye_outer[1], right_eye_outer[0]-left_eye_outer[0]), -pi/4), pi/4)
    #tilt_h = atan2(right_eye_outer[1]-left_eye_outer[1], right_eye_outer[0]-left_eye_outer[0])
    tilt_h = find_angle(left_eye_outer, right_eye_outer)

    tilt_lr = distance(left_eye_outer, face_center_line) / eye_dist - 0.5
    tilt_ud = distance(face_center_line, nose_top) / eye_dist-0.463-3*tilt_lr**2

    tilt_ud = min(max(tilt_ud, -0.02), 0.12)
    tilt_lr = min(max(tilt_lr, -0.15), 0.15)

    x = int(tilt_lr*150)
    y = 150-int(tilt_ud*625)
    #max(cos(20*(tilt_ud-0.026))+1.5, 0.4)
    #max(-50*(tilt_ud-0.025)**2+2, 1)
    r = 90

    pos = rotate(center, (center[0]+x, center[1]-y), tilt_h)

    cv2.circle(img, center, 10, (255, 0, 255), -1)
    cv2.circle(img, pos, r, (255, 0, 255), -1)

    # eyes

    eyes_center = rotate(pos, (pos[0], pos[1]+min(tilt_ud, 0.06)*600), tilt_h)

    left_eye_vdist = distance(point(face[159]), point(face[145]))/eye_dist
    right_eye_vdist = distance(point(face[386]), point(face[374]))/eye_dist

    #print(left_eye_vdist)

    cv2.circle(img, (eyes_center[0]+int(tilt_lr*150), eyes_center[1]), 5, (0, 0, 255), -1)

    eyes_center = (eyes_center[0]+int(tilt_lr*150), eyes_center[1])

    h = 40*sin(pi/8)
    w = 80*cos(pi/8)-abs(tilt_lr)*8/0.5

    p = tilt_lr+0.5

    l_angle = atan2(h, p*w)
    r_angle = atan2(h, (1-p)*w)

    l_dist = h/sin(l_angle)
    r_dist = h/sin(r_angle)

    left_eye_pos = move(eyes_center, l_dist, pi-l_angle-tilt_h)
    right_eye_pos = move(eyes_center, r_dist, r_angle-tilt_h)

    blink = 0.072

    if left_eye_vdist < blink:
        draw_closed_eye(move(left_eye_pos, 15, 2 * pi - tilt_h), 5 * pi / 6, tilt_h)
    else:
        draw_eye(left_eye_pos)

    if right_eye_vdist < blink:
        draw_closed_eye(move(right_eye_pos, 15, pi - tilt_h), pi / 6, tilt_h)
    else:
        draw_eye(right_eye_pos)



    #draw_eye(right_eye_pos)

    # mouth

    mouth_center = move(eyes_center, 30, -pi / 2 - tilt_h)

    lip_ud = distance(lip_u_center, lip_l_center)
    lip_lr = distance(lip_left, lip_right)

    mouth_h = int(min(lip_ud/ eye_dist, 0.4)*90)
    mouth_w = int(lip_lr * 60 / eye_dist)
    lip_l_r = min(length(shortest_dist(lip_left, lip_u_center, lip_l_center) / max(lip_lr, 0.01)), 1)
    lip_l = round(max(min(lip_l_r, 0.6), 0.4) * mouth_w)

    # closed mouth
    if mouth_h < 6:
        # shortest distance from lip upper center to horizontal line
        v = shortest_dist(lip_u_center, lip_left, lip_right)
        # get sign of y component
        direction = copysign(1, v[1])
        v_length = length(v)
        print(v_length)
        mouth_h = min(round(v_length*160*max(4*cos((max(min(2.5, v_length*direction), 0.2)-1)), 1)/eye_dist), 10)
        draw_closed_mouth(mouth_center, mouth_w, mouth_h, lip_l, tilt_h, direction)
    else:
        # open mouth
        lip_u_r = min(length(shortest_dist(lip_u_left, lip_left, lip_right) / max(lip_ud, 0.01)), 1)
        # print(lip_u_r, max(1.5*cos(8*(lip_u_r-0.38)), 1))

        draw_mouth(mouth_center, mouth_w, mouth_h, (lip_l, round(
            lip_u_r * max(1.8 * cos(8 * (min(lip_u_r, 0.6) - 0.3)), 1) * mouth_h)), tilt_h)

    #cv2.polylines(img, [np.array([move(mouth_center, 100, pi/2-tilt_h), mouth_center, move(mouth_center, 100, 3*pi/2-tilt_h)], dtype=np.int32)], False, (255, 0, 0), 2)

    cv2.circle(img, left_eye_outer, 5, (0, 0, 255), -1)
    cv2.circle(img, right_eye_outer, 5, (255, 0, 255), -1)


while cap.isOpened():
    s, img = cap.read()
    res = holistic.process(img)

    #img = np.ones((HEIGHT, WIDTH, 3))

    mpdraw.draw_landmarks(img, res.pose_landmarks, mpholistic.POSE_CONNECTIONS)
    #mpdraw.draw_landmarks(img, res.face_landmarks, mpholistic.FACE_CONNECTIONS)

    if res.face_landmarks:
        draw(res.face_landmarks.landmark, res.pose_landmarks.landmark)

    cv2.imshow('aaaaa', img)
    cv2.waitKey(1)
