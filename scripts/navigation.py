import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, 'python_utils')))

import rospy
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool, CommandTOL
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from panorama.srv import RequestMap, RequestMapResponse
from target_detection import detect_target
from kudrone_py_utils import *
import numpy as np
import math
import time
import cv_bridge

def wait_until_condition(condition, timeout):
    t_start = rospy.Time.now()
    while duration_to_sec(rospy.Time.now()-t_start) < timeout or timeout == -1:
        if condition():
            return True
    return False

class ROSArduPilot:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        rospy.Subscriber("/mavros/state", State, self.__state_callback)
        rospy.Subscriber("/mavros/global_position/local", Odometry, self.__position_callback)
        self.__setpoint_position_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.__set_mode_proxy = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.__arming_proxy = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.__takeoff_proxy = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)
        self.__land_proxy = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)
        wait_until_condition(lambda:hasattr(self, "connected"), -1)
        wait_until_condition(lambda:hasattr(self, "position"), -1)
    
    def travel_to_blocking(self, point, threshold = 1, timeout=-1):
        if not self.is_controllable():
            return False
        pose = PoseStamped()
        pose.pose.position.x = point[0,0]
        pose.pose.position.y = point[1,0]
        pose.pose.position.z = point[2,0]
        self.__setpoint_position_pub.publish(pose)
        t_start = rospy.Time.now()
        while duration_to_sec(rospy.Time.now()-t_start) < timeout or timeout == -1:
            if not self.is_controllable():
                return False
            elif self.distance_to(point) < threshold:
                return True
        return False

    def takeoff_blocking(self, altitude, threshold=1, timeout=-1):
        if not self.is_controllable():
            return False
        response = self.__takeoff_proxy(0,0,0,0,altitude)
        if not response.success:
            return False
        return wait_until_condition(lambda:(abs(self.position[2,0]-altitude)<threshold), timeout)
    
    def land_blocking(self, altitude=0, threshold=0.1, timeout=-1):
        if not self.is_controllable():
            return False
        response = self.__land_proxy(0,0,0,0,altitude)
        if not response.success:
            return False
        return wait_until_condition(lambda:(abs(self.position[2,0]-altitude)<threshold), timeout)


    def arm_disarm(self, arm=True, timeout=1):
        if self.armed == arm:
            return True
        result = self.__arming_proxy(arm)
        if not result.success:
            return False
        return wait_until_condition(lambda:self.armed==arm, timeout)

    def set_mode(self, mode, timeout=1):
        # print(mode)
        if self.mode == mode:
            return True
        response = self.__set_mode_proxy(0, mode)
        if response.mode_sent==False:
            return False
        return wait_until_condition(lambda:self.mode==mode, timeout)

    def distance_to(self, point):
        assert np.shape(point)==(3,1), "point must be a numpy array of shape (3,1)"
        diff = self.position-point
        return math.sqrt(np.matmul(diff.T, diff)[0,0])
    
    def is_controllable(self):
        return self.connected and self.armed and self.guided and self.mode == "GUIDED"

    def __state_callback(self, data: State):
        self.connected = data.connected
        self.armed = data.armed
        self.guided = data.guided
        self.manual_input = data.manual_input
        self.mode = data.mode
        self.system_status = data.system_status

    def __position_callback(self, data: PoseWithCovarianceStamped):
        position = data.pose.pose.position
        self.position = np.array([[position.x], [position.y], [position.z]])

if __name__ == "__main__":
    handle = ROSArduPilot("navigation")
    if not handle.set_mode("STABILIZE", -1):
        print("mode cant be STABILIZE")
        exit()
    if not handle.arm_disarm(True, -1):
        print("cant arm")
        exit()
    if not handle.set_mode("GUIDED", -1):
        print("mode cant be guided")
        exit()
    if not handle.takeoff_blocking(10):
        print("cant takeoff")
        exit()
    #Mapping pass
    print(handle.travel_to_blocking(np.array([[0], [50], [10]])))
    print(handle.travel_to_blocking(np.array([[100], [50], [10]])))
    print(handle.travel_to_blocking(np.array([[100], [-50], [10]])))
    print(handle.travel_to_blocking(np.array([[40], [-50], [10]])))
    #go to pool
    print(handle.travel_to_blocking(np.array([[20], [-20], [10]])))
    print(handle.travel_to_blocking(np.array([[20], [-20], [1]])))
    print(handle.travel_to_blocking(np.array([[20], [-20], [10]])))
    

    #get ready to go to target
    print(handle.travel_to_blocking(np.array([[0], [50], [10]])))
    print(handle.travel_to_blocking(np.array([[100], [50], [10]])))

    #find target
    print("Waiting for service")
    rospy.wait_for_service("/request_map")
    print("Found service")
    get_map = rospy.ServiceProxy("/request_map", RequestMap)
    response = get_map()
    local_pixels_to_global_meters_htm = np.array(response.map.local_pixels_to_global_meters_htm).reshape((4,4))
    print(local_pixels_to_global_meters_htm)
    bridge = cv_bridge.CvBridge()
    map_image = bridge.compressed_imgmsg_to_cv2(response.map.map_image)
    resolution = np.max(np.abs((local_pixels_to_global_meters_htm @ np.array([[0,0,0,1]]).T) - (local_pixels_to_global_meters_htm @ np.array([[1,0,0,1]]).T))[0:3,:])
    print(resolution)
    detected_targets = detect_target(map_image, min_radius=int(1/resolution), max_radius=int(2/resolution))
    if detected_targets is None:
        print("Cant find target")
        exit()
    for target in detected_targets[0, :]:
        print(target)
    target = np.array(detected_targets[0]).T
    target[2,0] = 0
    print(target)
    target = htm_multipliable_points_to_points(local_pixels_to_global_meters_htm @ points_to_htm_multipliable_points(target))
    print(target)
    target[2,0] = 10
    print(target)
    
    #travel to target
    print(handle.travel_to_blocking(target))
    #approach target
    target[2,0] = 1
    print(handle.travel_to_blocking(target))
    #depart target
    target[2,0] = 10
    print(handle.travel_to_blocking(target))

    print(handle.travel_to_blocking(np.array([[100], [-50], [10]])))
    print(handle.travel_to_blocking(np.array([[0], [-50], [10]])))
    print(handle.travel_to_blocking(np.array([[0], [0], [10]])))
    if not handle.land_blocking(0):
        print("cant land")
        exit()