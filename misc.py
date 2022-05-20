import numpy as np
from math import sin, cos, atan2, pi

HEIGHT = 720
WIDTH = 1280

def getxy(landmarks):
    global WIDTH, HEIGHT
    points = []
    for i in landmarks:
        points.append((int(i.x * WIDTH), int(i.y * HEIGHT)))
    return points

def point(landmark):
    global WIDTH, HEIGHT
    return (int(landmark.x * WIDTH), int(landmark.y * HEIGHT))

def midpoint(point1, point2):
    x = (point1[0]+point2[0])//2
    y = (point1[1]+point2[1])//2
    return (x, y)

# distance for xy only
def distance(point1, point2):
    #if all_dimensions:
        #return (((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)+((point1[2] - point2[2]) ** 2)) ** 0.5
    return (((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)) ** 0.5

# distance for all xyz
def landmark_distance(l1, l2):
    return (((l2.x-l1.x)**2)+((l2.y-l1.y)**2)+(((l2.z-l1.z)*0.78)**2))**0.5

def rotate(center, pos, angle) -> tuple:
    x = pos[0]-center[0]
    y = pos[1]-center[1]
    s = sin(angle)
    c = cos(angle)
    return (round(center[0]+x*c-y*s), round(center[1]+x*s+y*c))

def find_angle(p_left, p_right) -> float:
    return atan2(p_right[1] - p_left[1], p_right[0] - p_left[0])

def move(p, dist, angle) -> tuple:
    return (round(p[0]+dist*cos(angle)), round(p[1]-dist*sin(angle)))

def length(v):
    #print(v, 'aaaaa')
    return (v[0]**2+v[1]**2)**0.5

def dot(v1, v2):
    t = 0
    for i, c in zip(v1, v2):
        t += i*c
    return t

def shortest_dist(p, a, b):
    a = np.array(a, dtype=np.float16)
    b = np.array(b, dtype=np.float16)
    p = np.array(p, dtype=np.float16)
    ba = a-b
    t = ((p[0] - b[0]) * ba[0] + (p[1] - b[1]) * ba[1]) / max(dot(ba, ba), 0.01)
    v = b + t * ba-p
    return v
    #print(v)
    #return length(v)
    #print((pa[0] ** 2 + pa[1] ** 2) ** 0.5 / distance(lip_u_center, lip_l_center))
