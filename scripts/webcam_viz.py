#!/usr/bin/env python3
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
from scipy.interpolate import griddata

from mpl_toolkits.mplot3d import Axes3D

import time
import pickle

IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_HORIZONTAL_FOV = 1.2
IMG_START_POINTS = []

cam_pos = []
cam_rotation = []
cam_image = []



xy_bgr_total = np.array([[],[],[],[],[]])

class ImageAppend:
    def __init__(this, width, height, step = 0.4, depth = 3):
        this.step = step
        this.width = width
        this.height = height
        this.depth = depth
        this.image = np.zeros((height, width, depth))

    def updateImage(this, new_img):
        (img_height, img_width, img_depth) = np.shape(new_img)
        this.width = img_width
        this.height = img_height
        this.depth = img_depth
        this.image = new_img

    def pixel_to_origin_coords(this, pixel_values):
        return np.array([]).T
    
    def append(this, img, corner_pixel_values):
        corner_pixel_values = np.round(corner_pixel_values).astype(np.int)
        (img_height, img_width, _) = np.shape(img)
        x_min_img = np.min(corner_pixel_values.T[0])
        x_max_img = np.max(corner_pixel_values.T[0])
        y_min_img = np.min(corner_pixel_values.T[1])
        y_max_img = np.max(corner_pixel_values.T[1])
        print(x_min_img, y_min_img)

        new_width = max(this.width, x_max_img) - min(0, x_min_img)
        new_height = max(this.height, y_max_img) - min(0, y_min_img)
        new_img = np.zeros((new_height, new_width, this.depth))
        
        new_origin = np.array([[min(0, x_min_img), min(0, y_min_img)]], dtype=np.int)

        corner_pixel_values = (corner_pixel_values.T - new_origin.T).T
        
        new_img[-new_origin[0][1]:this.height-new_origin[0][1],-new_origin[0][0]:this.width-new_origin[0][0]] = this.image[:,:]
        
        x_min_img = np.min(corner_pixel_values.T[0])
        x_max_img = np.max(corner_pixel_values.T[0])
        y_min_img = np.min(corner_pixel_values.T[1])
        y_max_img = np.max(corner_pixel_values.T[1])

        new_img[y_min_img:y_min_img+img_height, x_min_img:x_min_img+img_width] = img[:, :]

        this.updateImage(new_img)

        


def cartesian_cross_product(x,y):
    cross_product = np.transpose([np.tile(x, len(y)),np.repeat(y,len(x))])
    return cross_product

def calculateCamImgInitialPos(width, height, horizontal_fov):
    focal_len = width/2/m.tan(horizontal_fov/2)
    y = np.array([height/2, -(height/2-1)]).T
    x = np.array([-width/2, width/2-1]).T
    return np.vstack([cartesian_cross_product(x, y).T, -focal_len*np.ones((4))])

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

def sampleImg(xy_bgr, step=0.1):

    projected_pts = xy_bgr[0:2]
    img_matrix = xy_bgr[2:5]
    x_min = np.min(projected_pts[0])
    x_max = np.max(projected_pts[0])

    y_min = np.min(projected_pts[1])
    y_max = np.max(projected_pts[1])
    
    y = np.arange(y_min, y_max, step)
    x = np.arange(x_min, x_max, step)
    query_pts = cartesian_cross_product(x, y)

    height = np.shape(y)[0]
    width = np.shape(x)[0]

    #k-d tree (slightly less than 4 hz)
    # tree = KDTree(projected_pts.T)
    # dist, ind = tree.query(query_pts, k=1)
    # b = np.take_along_axis(img_matrix[:,0], ind[:,0], 0).reshape((1, height*width))
    # g = np.take_along_axis(img_matrix[:,1], ind[:,0], 0).reshape((1, height*width))
    # r = np.take_along_axis(img_matrix[:,2], ind[:,0], 0).reshape((1, height*width))
    # new_img = np.vstack([b, g, r]).T.reshape((height, width, 3))

    #scipy.interpolate.griddata (less than 1 hz)
    # method = "nearest"
    # b = griddata(projected_pts.T, img_matrix[:,0], query_pts, method=method)
    # g = griddata(projected_pts.T, img_matrix[:,1], query_pts, method=method)
    # r = griddata(projected_pts.T, img_matrix[:,2], query_pts, method=method)
    # new_img = np.vstack([b, g, r]).T.reshape((height, width, 3))

    x_range = x_max-x_min
    y_range = y_max-y_min
    new_img = np.zeros((height, width, 3), dtype=np.uint8)

    projected_pts = np.around(projected_pts/step)*step

    projected_pts_ind_x = (projected_pts[0]-x_min)/x_range*width
    projected_pts_ind_y = (projected_pts[1]-y_min)/y_range*height

    projected_pts_ind_x[projected_pts_ind_x<0] = 0
    projected_pts_ind_x[projected_pts_ind_x>width-1] = width-1

    projected_pts_ind_y[projected_pts_ind_y<0] = 0
    projected_pts_ind_y[projected_pts_ind_y>height-1] = height-1

    projected_pts_ind = np.vstack([projected_pts_ind_x, projected_pts_ind_y]).astype(int)
    # projected_pts_ind = (projected_pts_ind_x + projected_pts_ind_y*height).astype(int)

    # new_img = np.take_along_axis(img_matrix, projected_pts_ind.T, axis=0)
    
    # projected_pts_ind.h

    new_img[projected_pts_ind.T[:, 1], projected_pts_ind.T[:, 0]] = img_matrix.T
    new_img = new_img.reshape((height, width, 3))

    return new_img

def projected_pts_to_pixel(projected_points, step=0.4):
    # projected_points = projected_points[::-1,:]
    
    x_min = np.min(projected_points[0])
    x_max = np.max(projected_points[0])

    y_min = np.min(projected_points[1])
    y_max = np.max(projected_points[1])
    print(x_min, y_min)
    projected_points_abs = np.copy(projected_points)

    projected_points -= [[x_min], [y_max]]
    
    projected_points[1] = -projected_points[1]
    projected_points_abs[1] = -projected_points_abs[1]

    pixel_vals = projected_points/step
    pixel_vals_abs = projected_points_abs/step
    
    pixel_vals_abs += [[IMG_WIDTH], [IMG_HEIGHT]]

    return pixel_vals.astype(np.float32).T, pixel_vals_abs.astype(np.float32).T

def posCallback(data):
    global times
    global cam_pos
    global cam_rotation
    times = []
    pose = data.pose.pose
    point = pose.position
    quat = pose.orientation
    cam_pos = np.array([[point.x], [point.y], [point.z]])
    cam_rotation = np.array(quaternion_rotation_matrix(np.array([[quat.w], [quat.x], [quat.y], [quat.z]]))).reshape((3,3))
    
    times.append(time.time())

    camera_points = np.matmul(cam_rotation, IMG_START_POINTS) + cam_pos
    # print(camera_points)
    projected_points = project_points(camera_points, cam_pos)
    times.append(time.time())


    #projected pts to pixel values
    from_pts = np.float32([[0,0], [IMG_WIDTH-1,0], [0, IMG_HEIGHT-1], [IMG_WIDTH-1, IMG_HEIGHT-1]])
    to_pts, to_pts_abs = projected_pts_to_pixel(projected_points)


    perspective_matrix = cv2.getPerspectiveTransform(from_pts, to_pts_abs)
    
    width = m.ceil(np.max(to_pts[:,0]))
    height = m.ceil(np.max(to_pts[:,1]))
    
    perspective_img = cv2.warpPerspective(cam_image, perspective_matrix, (IMG_WIDTH*2, IMG_HEIGHT*2))

    # img_append.append(perspective_img, to_pts_abs)
    # times.append(time.time())

    cv2.imshow("img", perspective_img)
    # cv2.imwrite("img.jpg", img_append.image)
    cv2.waitKey(1)

    # # hue = np.array(hsv[:,:,0]).reshape((1, IMG_HEIGHT*IMG_WIDTH))
    # # times.append(time.time())

    # # fields = [PointField('x', 0, PointField.FLOAT32, 1),
    # #           PointField('y', 4, PointField.FLOAT32, 1),
    # #           PointField('z', 8, PointField.FLOAT32, 1),
    # #           PointField('hue', 12, PointField.FLOAT32, 1)]

    # # header = std_msgs.msg.Header()
    # # header.frame_id = "map"
    # # header.stamp = rospy.Time.now()

    # # points = np.vstack((projected_points, np.zeros((1, np.shape(projected_points)[1])), hue)).T

    # # pc2 = point_cloud2.create_cloud(header, fields, points[::10])
    # # pub.publish(pc2)
    
    # # global projected_points_total
    # # global img_matrix_total

    # img_matrix = np.array(cam_image).reshape((IMG_HEIGHT*IMG_WIDTH, 3)).T
    # xy_bgr = np.vstack([projected_points, img_matrix])
    # if cam_pos[2] > 1:
    #     global xy_bgr_total
    #     xy_bgr_total = np.hstack([xy_bgr_total, xy_bgr])
    #     print(np.shape(xy_bgr_total))
    # new_img = sampleImg(xy_bgr, step=0.1)
    
    # cv2.imshow("img", new_img)
    # cv2.waitKey(1)

    # pickle_file = "scan.pickle"
    # if np.shape(xy_bgr_total)[1] % IMG_HEIGHT*IMG_WIDTH*5 == 0:
    #     with open(pickle_file, "wb") as file:
    #         pickle.dump(xy_bgr_total, file)
    # print(np.shape(new_img))
    # times.append(time.time())
    print([times[i+1] - times[i] for i in range(len(times)-1)], times[-1]-times[0])

    # # fig = plt.figure()

    # # ax = fig.add_subplot(111, projection='3d')
    
    # # ax.scatter(camera_points[0],camera_points[1],camera_points[2])

    # # plt.show()

    # #<node name="webcam_viz" pkg="opencv_drone" type="webcam_viz.py"/>
    # # rospy.loginfo(cam_rotation)
    
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

    rot = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])

    IMG_START_POINTS = np.matmul(rot, calculateCamImgInitialPos(IMG_WIDTH, IMG_HEIGHT, IMG_HORIZONTAL_FOV))
    # IMG_START_POINTS = calculateCamImgInitialPos(IMG_WIDTH, IMG_HEIGHT, IMG_HORIZONTAL_FOV)
    print(IMG_START_POINTS)
    global img_append
    img_append = ImageAppend(IMG_WIDTH*2, IMG_HEIGHT*2)
    rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except rospy.ROSInterruptException:
        pass
