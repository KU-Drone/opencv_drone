#!/usr/bin/env
import rospy

import std_msgs.msg
import sensor_msgs.msg
from sensor_msgs import point_cloud2  
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointField
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2

import cv2
import numpy as np
import math as m
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from mpl_toolkits.mplot3d import Axes3D

import time

IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_HORIZONTAL_FOV = 1.2
IMG_START_POINTS = []

cam_pos = []
cam_rotation = []
cam_image = []

def cartesian_cross_product(x,y):
    cross_product = np.transpose([np.tile(x, len(y)),np.repeat(y,len(x))])
    return cross_product


def calculateCamImgInitialPos(width, height, horizontal_fov):
    focal_len = width/2/m.tan(horizontal_fov/2)
    y = np.arange(height/2, -height/2, -1)
    x = np.arange(-width/2, width/2, 1)
    return np.vstack([cartesian_cross_product(x, y).T, -focal_len*np.ones((width*height))])

def project_points(camera_points, camera_position):
    xc = camera_position[0][0]
    yc = camera_position[1][0]
    zc = camera_position[2][0]

    denom = zc - camera_points[2]

    transform1 = np.array([[zc, 0, -xc],
                           [0, zc, -yc]])

    camera_points = np.matmul(transform1, camera_points)

    return np.divide(camera_points, [denom, denom])

def imgCallback(data):
    global cam_image
    bridge = CvBridge()
    cam_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

def sampleImg(projected_pts, raw_img, step=0.1):
    tree = KDTree(projected_pts.T)

    x_min = np.min(projected_pts[0])
    x_max = np.max(projected_pts[0])

    y_min = np.min(projected_pts[1])
    y_max = np.max(projected_pts[1])
    
    y = np.arange(y_min, y_max, step)
    x = np.arange(x_min, x_max, step)
    query_pts = cartesian_cross_product(x, y)
    times.append(time.time())
    # print(query_pts)
    img_matrix = np.array(cam_image).reshape((IMG_HEIGHT*IMG_WIDTH, 3))

    height = np.shape(y)[0]
    width = np.shape(x)[0]

    dist, ind = tree.query(query_pts, k=1)
    # img_matrix = img_matrix[ind]
    # new_img = np.zeros((height, width, 3), dtype = np.uint8)
    # for i in range(np.shape(ind)[0]):
    #     new_img[i//width][i%width] = img_matrix[ind[i]]
    # print(np.shape(img_matrix[:,0]))
    b = np.take_along_axis(img_matrix[:,0], ind[:,0], 0).reshape((1, height*width))
    g = np.take_along_axis(img_matrix[:,1], ind[:,0], 0).reshape((1, height*width))
    r = np.take_along_axis(img_matrix[:,2], ind[:,0], 0).reshape((1, height*width))
    new_img = np.vstack([b, g, r])
    print(np.shape(new_img))
    # print(np.shape(b))
    # print(new_img)
    new_img = new_img.T.reshape((height, width, 3))
    # print(new_img)

    return new_img

def posCallback(data):
    global times
    times = []
    global cam_pos
    global cam_rotation
    pose = data.pose.pose
    point = pose.position
    quat = pose.orientation
    cam_pos = np.array([[point.x], [point.y], [point.z]])
    cam_rotation = np.array(quaternion_rotation_matrix(np.array([[quat.w], [quat.x], [quat.y], [quat.z]]))).reshape((3,3))
    
    times.append(time.time())
    hsv = cv2.cvtColor(cam_image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("image", cv2_image)
    # cv2.waitKey(1)

    camera_points = np.matmul(cam_rotation, IMG_START_POINTS) + cam_pos
    # print(camera_points)
    projected_points = project_points(camera_points, cam_pos)
    times.append(time.time())

    # hue = np.array(hsv[:,:,0]).reshape((1, IMG_HEIGHT*IMG_WIDTH))
    # times.append(time.time())

    # fields = [PointField('x', 0, PointField.FLOAT32, 1),
    #           PointField('y', 4, PointField.FLOAT32, 1),
    #           PointField('z', 8, PointField.FLOAT32, 1),
    #           PointField('hue', 12, PointField.FLOAT32, 1)]

    # header = std_msgs.msg.Header()
    # header.frame_id = "map"
    # header.stamp = rospy.Time.now()

    # points = np.vstack((projected_points, np.zeros((1, np.shape(projected_points)[1])), hue)).T

    # pc2 = point_cloud2.create_cloud(header, fields, points[::10])
    # pub.publish(pc2)
    new_img = sampleImg(projected_points, cam_image, step=0.2)
    
    cv2.imshow("img", new_img)
    cv2.waitKey(1)
    print(np.shape(new_img))

    times.append(time.time())
    print([times[i+1] - times[i] for i in range(len(times)-1)], times[-1]-times[0])

    # fig = plt.figure()

    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(camera_points[0],camera_points[1],camera_points[2])

    # plt.show()

    #<node name="webcam_viz" pkg="opencv_drone" type="webcam_viz.py"/>
    # rospy.loginfo(cam_rotation)
    
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def node():
    rospy.init_node("webcam_viz", anonymous=False)
    rospy.Subscriber("/webcam/image_raw", Image, imgCallback)
    rospy.Subscriber("/mavros/global_position/local", Odometry, posCallback)
    global pub
    pub = rospy.Publisher('points', PointCloud2, queue_size=10)
    global IMG_START_POINTS
    IMG_START_POINTS = calculateCamImgInitialPos(IMG_WIDTH, IMG_HEIGHT, IMG_HORIZONTAL_FOV)
    print(IMG_START_POINTS)
    rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except rospy.ROSInterruptException:
        pass
