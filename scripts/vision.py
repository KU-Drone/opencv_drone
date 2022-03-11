#!/usr/bin/env python3
import rospy

import sensor_msgs.msg
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import cv2
import numpy as np
import math as m

import sys
import time

IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_HORIZONTAL_FOV = 1.2
IMG_START_POINTS = []

cam_pos = []
cam_rotation = []
cam_image = []

first_time = 2

step = 0.1

img_path = "img.jpg"

class ImageAppend:
    def __init__(this, width, height, step = 0.2, depth = 3):
        this.step = step
        this.width = width
        this.height = height
        this.depth = depth
        this.map_corner_coords = np.array([[0, 0]], dtype=np.int)  #coordinates of where the map top left corner is in local pixel coordinate frame
        this.image = np.zeros((height, width, depth))

    def updateImage(this, new_img):
        (img_height, img_width, img_depth) = np.shape(new_img)
        this.width = img_width
        this.height = img_height
        this.depth = img_depth
        this.image = new_img

    # def pixel_to_origin_coords(this, pixel_values):
    #     return np.array([]).T

    def local_meter_to_local_pixel_coords(this, local_meter_coords):
        local_meter_coords_temp = np.copy(local_meter_coords)

        local_meter_coords_temp[1] = -local_meter_coords_temp[1]

        local_pixel_coords = local_meter_coords_temp/this.step

        return local_pixel_coords.astype(np.float32)

    def project(this, img, projected_points):
        to_pts_abs = img_append.local_meter_to_local_pixel_coords(projected_points)
        corner_pixel_values = to_pts_abs.T

        #round the coordinates of the corners which are in home center pixel coordinate frame
        corner_pixel_values = np.round(corner_pixel_values).astype(np.int)
        (img_height, img_width, _) = np.shape(img)
        
        #get boundaries of the image to add
        x_min_img = np.min(corner_pixel_values.T[0])
        x_max_img = np.max(corner_pixel_values.T[0])
        y_min_img = np.min(corner_pixel_values.T[1])
        y_max_img = np.max(corner_pixel_values.T[1])
        
        #set the image width to span from the lowest x to the highest. Same with the height
        new_width = max(this.width+this.map_corner_coords[0][0], x_max_img+1) - min(this.map_corner_coords[0][0], x_min_img)
        new_height = max(this.height+this.map_corner_coords[0][1], y_max_img+1) - min(this.map_corner_coords[0][1], y_min_img)

        #initialise empty image to copy the current image into. These are in the form of the updated image pixel coordinates. 
        old_img_new_index = np.zeros((new_height, new_width, this.depth))
        
        #save the old map corner coordinates
        old_map_corner_coords = np.copy(this.map_corner_coords)
        #new map corner coordinates are the lower of the x and y values of old map_corner_coords and top left corner of image to stitch
        this.map_corner_coords = np.array([[min(this.map_corner_coords[0][0], x_min_img), min(this.map_corner_coords[0][1], y_min_img)]], dtype=np.int)

        #how much to offset new image
        offset = this.map_corner_coords
        #how much to offset old image
        offset_old = this.map_corner_coords-old_map_corner_coords

        #copy every pixel from old image into the empty old_img_new_index picture which has its map_corner_coords at the new map_corner_coords
        old_img_new_index[-offset_old[0][1]:this.height-offset_old[0][1],-offset_old[0][0]:this.width-offset_old[0][0]] = this.image[:,:]
        old_img_new_index = old_img_new_index.astype(np.float32)#needed for opencv for some reason

        #corners of the input image
        from_pts = np.float32([[0,0], [img_width-1,0], [0, img_height-1], [img_width-1, img_height-1]])
        #where the corners go (in pixel coordinates)
        to_pts = ((corner_pixel_values.T - this.map_corner_coords.T).T).astype(np.float32)

        if not x_max_img == x_min_img and not y_max_img == y_min_img:
            #project the image only if the pixel values arent the same. the coordinate space is the same as old_img_new_index
            perspective_matrix = cv2.getPerspectiveTransform(from_pts, to_pts)
            new_img_new_index = cv2.warpPerspective(img, perspective_matrix, (new_width,new_height))
        else:
            new_img_new_index = np.zeros((new_height, new_width, this.depth)).astype(np.float32)

        #create a mask to black out the region where the new image will fit in the old image
        new_img_new_index_gray = cv2.cvtColor(new_img_new_index, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(new_img_new_index_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask.astype(np.uint8))

        #black out area where the new image will fit in the old image
        old_img_new_index = cv2.bitwise_and(old_img_new_index, old_img_new_index, mask=mask_inv.astype(np.uint8))

        #stitch images
        ret = old_img_new_index + new_img_new_index

        this.updateImage(ret)

def cartesian_cross_product(x,y):
    cross_product = np.transpose([np.tile(x, len(y)),np.repeat(y,len(x))])
    return cross_product

def calculateCamImgInitialPos(width, height, horizontal_fov):
    focal_len = width/2/m.tan(horizontal_fov/2)
    y = np.array([height/2, -(height/2)]).T
    x = np.array([-width/2, width/2]).T
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
    
    projected_points = project_points(camera_points, cam_pos)
    
    times.append(time.time())
    img_append.project(cam_image, projected_points)

    times.append(time.time())

    cv2.imwrite(img_path, img_append.image)

    rospy.loginfo(f"{[(times[i+1] - times[i])/(times[-1]-times[0])*100 for i in range(len(times)-1)]} {1/(times[-1]-times[0])}")

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
    global IMG_START_POINTS
    global img_append
    global img_path

    rot = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])

    IMG_START_POINTS = np.matmul(rot, calculateCamImgInitialPos(IMG_WIDTH, IMG_HEIGHT, IMG_HORIZONTAL_FOV))

    rospy.loginfo(IMG_START_POINTS)
    
    img_append = ImageAppend(IMG_WIDTH//2, IMG_HEIGHT//2, step=step)
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        rospy.logdebug(img_path)
    rospy.spin()

if __name__ == "__main__":
    try:
        node()
    except rospy.ROSInterruptException:
        pass
