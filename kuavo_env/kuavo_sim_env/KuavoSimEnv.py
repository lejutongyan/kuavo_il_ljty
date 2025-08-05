#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float32MultiArray,Float64MultiArray,Int32
from std_msgs.msg import Bool
import cv2
import gymnasium as gym
import time
from env.KuavoBaseRosEnv import KuavoBaseRosEnv

class KuavoSimEnv(KuavoSimEnv):

    def compute_reward(self):
        return 0