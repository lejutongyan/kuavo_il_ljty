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
from kuavo_env.config.config_kuavo_env import load_config
from kuavo_msgs.msg._sensorsData import sensorsData
from kuavo_msgs.msg._robotHandPosition import robotHandPosition
from kuavo_msgs.msg._lejuClawState import lejuClawState
from kuavo_msgs.msg._lejuClawCommand import lejuClawCommand
from kuavo_msgs.srv._changeArmCtrlMode import changeArmCtrlModeRequest, changeArmCtrlMode
from kuavo_msgs.msg._robotHeadMotionData import robotHeadMotionData

class KuavoBaseRosEnv(gym.Env):

    def __init__(self):
        config_kuavo_env = load_config()
        self.real = config_kuavo_env.real
        self.ros_rate = config_kuavo_env.ros_rate
        self.control_mode = config_kuavo_env.control_mode
        self.image_size = config_kuavo_env.image_size
        self.only_arm = config_kuavo_env.only_arm
        self.eef_type = config_kuavo_env.eef_type
        self.which_arm = config_kuavo_env.which_arm
        self.qiangnao_dof_needed = config_kuavo_env.qiangnao_dof_needed
        self.leju_claw_dof_needed = config_kuavo_env.leju_claw_dof_needed
        self.arm_init = config_kuavo_env.arm_init
        self.slice_robot = config_kuavo_env.slice_robot
        self.qiangnao_slice = config_kuavo_env.qiangnao_slice
        self.claw_slice = config_kuavo_env.claw_slice
        self.is_binary = config_kuavo_env.is_binary

        self.arm_min = config_kuavo_env.arm_min
        self.arm_max = config_kuavo_env.arm_max
        self.eef_min = config_kuavo_env.eef_min
        self.eef_max = config_kuavo_env.eef_max

        self.bridge = CvBridge()
        rospy.init_node('kuavo_base_ros_env', anonymous=True)
        self.rate = rospy.Rate(self.ros_rate)

        self.pub_head_joint = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
        
        if self.only_arm:
            if not self.real:
                # ROS subscribers
                rospy.Subscriber("/cam_h/color/image_raw/compressed", CompressedImage, self.cam_h_callback)
                rospy.Subscriber("/cam_l/color/image_raw/compressed", CompressedImage, self.cam_l_callback)
                rospy.Subscriber("/cam_r/color/image_raw/compressed", CompressedImage, self.cam_r_callback)
                rospy.Subscriber("/F_state", JointState, self.F_state_callback)

                if self.eef_type == 'leju_claw':
                    self.pub_claw_joint = rospy.Publisher('/claw_cmd', JointState, queue_size=10)
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
                
                self.pub_change_arm_ctrl_mode = rospy.Publisher('/lerobot/change_arm_ctrl_mode', Int32, queue_size=10)
            else:
                # ROS subscribers
                rospy.Subscriber("/cam_h/color/image_raw/compressed", CompressedImage, self.cam_h_callback)
                rospy.Subscriber("/cam_l/color/image_raw/compressed", CompressedImage, self.cam_l_callback)
                rospy.Subscriber("/cam_r/color/image_raw/compressed", CompressedImage, self.cam_r_callback)
                rospy.Subscriber("/sensors_data_raw", sensorsData, self.joint_state_callback)
                
                if self.eef_type == 'leju_claw':
                    rospy.Subscriber("/leju_claw_state", lejuClawState, self.claw_state_callback)
                    self.pub_claw_joint = rospy.Publisher('/leju_claw_command', lejuClawCommand, queue_size=10)
                elif self.eef_type == 'qiangnao':
                    # rospy.Subscriber("/control_robot_hand_position_state", robotHandPosition, self.qiangnao_state_callback)
                    rospy.Subscriber("/dexhand/state", JointState, self.qiangnao_state_callback)
                    self.pub_qiangnao_joint = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=10)

                self.pub_change_arm_ctrl_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            
            # ROS publishers
            self.pub_arm_joint = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)
            
            # Initialize action space based on control mode
            if self.control_mode == 'joint':
                if self.which_arm == 'both':
                    self.arm_joint_dim = 14
                    action_low = np.concatenate((self.arm_min[:7], self.eef_min, self.arm_min[7:14], self.eef_min), axis=0)
                    action_high = np.concatenate((self.arm_max[:7], self.eef_max, self.arm_max[7:14], self.eef_max), axis=0)
                    state_low = np.concatenate((self.arm_min[:7], self.eef_min, self.arm_min[7:14], self.eef_min), axis=0)
                    state_high = np.concatenate((self.arm_max[:7], self.eef_max, self.arm_max[7:14], self.eef_max), axis=0)
                elif self.which_arm == 'left':
                    self.arm_joint_dim = 7
                    action_low = np.concatenate((self.arm_min[:7], self.eef_min), axis=0)
                    action_high = np.concatenate((self.arm_max[:7], self.eef_max), axis=0)
                    state_low = np.concatenate((self.arm_min[:7], self.eef_min), axis=0)
                    state_high = np.concatenate((self.arm_max[:7], self.eef_max), axis=0)
                elif self.which_arm == 'right':
                    self.arm_joint_dim = 7
                    action_low = np.concatenate((self.arm_min[7:], self.eef_min), axis=0)
                    action_high = np.concatenate((self.arm_max[7:], self.eef_max), axis=0)
                    state_low = np.concatenate((self.arm_min[7:], self.eef_min), axis=0)
                    state_high = np.concatenate((self.arm_max[7:], self.eef_max), axis=0)
            elif self.control_mode == 'eef':
                raise KeyError("control_mode = 'eef' is not supported!")
                # self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))

            if self.eef_type == 'leju_claw':
                if self.leju_claw_dof_needed == 1:
                    if self.which_arm == 'both':
                        self.claw_joint_dim = 2
                    elif self.which_arm == 'left':
                        self.claw_joint_dim = 1
                    elif self.which_arm == 'right':
                        self.claw_joint_dim = 1
                self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.arm_joint_dim + self.claw_joint_dim,), dtype=np.float64)

            elif self.eef_type == 'qiangnao':
                if self.qiangnao_dof_needed == 1:
                    if self.which_arm == 'both':
                        self.qiangnao_joint_dim = 2
                    elif self.which_arm == 'left':
                        self.qiangnao_joint_dim = 1
                    elif self.which_arm == 'right':
                        self.qiangnao_joint_dim = 1
                self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.arm_joint_dim + self.qiangnao_joint_dim,), dtype=np.float64)
            
            # Observation space
            if self.which_arm == 'both':
                if self.eef_type == 'leju_claw':
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.Box(low=state_low, high=state_high, shape=(self.arm_joint_dim + self.claw_joint_dim,)),
                        "images": gym.spaces.Dict({
                            "cam_h": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_l": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_r": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                        })
                    })
                elif self.eef_type == 'qiangnao':
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.Box(low=state_low, high=state_high, shape=(self.arm_joint_dim + self.qiangnao_joint_dim,)),
                        "images": gym.spaces.Dict({
                            "cam_h": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_l": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_r": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                        })
                    })
            elif self.which_arm == 'left':
                if self.eef_type == 'leju_claw':
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.Box(low=state_low, high=state_high, shape=(self.arm_joint_dim + self.claw_joint_dim,)),
                        "images": gym.spaces.Dict({
                            "cam_h": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_l": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                        })
                    })
                elif self.eef_type == 'qiangnao':
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.Box(low=state_low, high=state_high, shape=(self.arm_joint_dim + self.qiangnao_joint_dim,)),
                        "images": gym.spaces.Dict({
                            "cam_h": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_l": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                        })
                    })
            elif self.which_arm == 'right':
                if self.eef_type == 'leju_claw':
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.Box(low=state_low, high=state_high, shape=(self.arm_joint_dim + self.claw_joint_dim,)),
                        "images": gym.spaces.Dict({
                            "cam_h": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_l": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_r": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                        })
                    })
                elif self.eef_type == 'qiangnao':
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.Box(low=state_low, high=state_high, shape=(self.arm_joint_dim + self.qiangnao_joint_dim,)),
                        "images": gym.spaces.Dict({
                            "cam_h": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                            "cam_r": gym.spaces.Box(0, 255, shape=self.image_size, dtype=np.uint8),
                        })
                    })
            else:
                raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")
            # Initialize state variables
            self.cam_h_img = None
            self.cam_l_img = None
            self.cam_r_img = None
            self.state = None
            self.start_state = None
        else:
            raise KeyError("only_arm = False is not supported!")

    def compute_reward(self):
        return 0

    def reset(self, **kwargs):
        if not self.real:
            msg = Int32()
            msg.data = 2
            for _ in range(500):  # Loop to send multiple reset commands
                time.sleep(0.001)
                self.pub_change_arm_ctrl_mode.publish(msg)
        else:
            msg = changeArmCtrlModeRequest()
            msg.control_mode = 2
            for i in range(10):  # Loop to send multiple reset commands
                time.sleep(0.001)
                print(f"reset control mode loop {i}")
                self.pub_change_arm_ctrl_mode(msg)
        msg = robotHeadMotionData()
        msg.joint_data = [0,25]
        for i in range(10):
            self.pub_head_joint.publish(msg)
            time.sleep(0.001)
            print(f"reset head loop {i}")

        if self.real:
            if self.which_arm == 'both':
                if self.eef_type == 'qiangnao':
                    msg = robotHandPosition()
                    msg.left_hand_position = [0,100,0,0,0,0]
                    msg.right_hand_position = [0,100,0,0,0,0]
                    for _ in range(10):
                        self.pub_qiangnao_joint.publish(msg)
                        time.sleep(0.1)
                elif self.eef_type == 'leju_claw':
                    raise KeyError("leju_claw is not supported!")
            elif self.which_arm == 'left':
                if self.eef_type == 'qiangnao':
                    msg = robotHandPosition()
                    msg.left_hand_position = [0,100,0,0,0,0]
                    msg.right_hand_position = [0,0,0,0,0,0]
                    for _ in range(10):
                        self.pub_qiangnao_joint.publish(msg)
                        time.sleep(0.1)
                elif self.eef_type == 'leju_claw':
                    raise KeyError("leju_claw is not supported!")
            elif self.which_arm == 'right':
                if self.eef_type == 'qiangnao':
                    msg = robotHandPosition()
                    msg.left_hand_position = [0,0,0,0,0,0]
                    msg.right_hand_position = [0,100,0,0,0,0]
                    print("publish qiangnao joint")
                    for _ in range(10):
                        self.pub_qiangnao_joint.publish(msg)
                        time.sleep(0.1)
                elif self.eef_type == 'leju_claw':
                    raise KeyError("leju_claw is not supported!")
            else:
                raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")

        average_num = 10
        for i in range(average_num):
            state = self.get_obs()
            if i==0:
                self.start_state = state["state"]
            else:
                self.start_state += state["state"]
            time.sleep(0.001)
        self.start_state = self.start_state / average_num

        obs = self.get_obs()
        return obs, {}

    def step(self, action):
        # Execute action
        print("joint_q:",self.joint_q/np.pi*180)
        if self.real:
            if self.which_arm == 'both':
                action[:7] = action[:7]/np.pi*180
                action[8:15] = action[8:15]/np.pi*180
            elif self.which_arm == 'left' or self.which_arm == 'right':
                action[:7] = action[:7] / np.pi * 180
            # action = action/np.pi*180

        print("action:",action)
        action = np.clip(action, self.action_space.low, self.action_space.high) # 限制动作范围
        print("clip action:",action)
        self.exec_action(action)
        self.rate.sleep()
        
        # Get new observation
        obs = self.get_obs()
        
        # Simplified reward and termination
        reward = self.compute_reward()
        done = False
        return obs, reward, done, False, {}

    def exec_action(self, action):
        if self.only_arm:
            if not self.real:
                if self.which_arm == 'both':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        arm_msg = JointState()
                        arm_msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
                        arm_msg.header.stamp = rospy.Time.now()
                        arm_msg.position = np.concatenate((action[:7], action[8:15]), axis=0)
                        self.pub_arm_joint.publish(arm_msg)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:                        
                            claw_msg = JointState()
                            claw_msg.position = np.concatenate(([action[7]*100], [action[15]*100]), axis=0)
                            self.pub_claw_joint.publish(claw_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        raise KeyError("qiangnao is not supported!")
                elif self.which_arm == 'left':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        arm_msg = JointState()
                        arm_msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
                        arm_msg.header.stamp = rospy.Time.now()
                        arm_msg.position = np.concatenate((action[:7], self.arm_init[7:14]), axis=0)
                        self.pub_arm_joint.publish(arm_msg)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            claw_msg = JointState()
                            claw_msg.position = np.concatenate(([action[7]*100], [0]), axis=0)
                            self.pub_claw_joint.publish(claw_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        raise KeyError("qiangnao is not supported!")
                elif self.which_arm == 'right':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        arm_msg = JointState()
                        arm_msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
                        arm_msg.header.stamp = rospy.Time.now()
                        arm_msg.position = np.concatenate((self.arm_init[:7],action[:7]), axis=0)
                        self.pub_arm_joint.publish(arm_msg)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            claw_msg = JointState()
                            claw_msg.position = np.concatenate(([0],[action[7]*100]), axis=0)
                            self.pub_claw_joint.publish(claw_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        raise KeyError("qiangnao is not supported!")
                else:
                    raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")
            else:
                if self.which_arm == 'both':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        arm_msg = JointState()
                        arm_msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
                        arm_msg.header.stamp = rospy.Time.now()
                        arm_msg.position = np.concatenate((action[:7], action[8:15]), axis=0)
                        self.pub_arm_joint.publish(arm_msg)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            claw_msg = lejuClawCommand()
                            claw_msg.data.position = np.concatenate(([action[7]], [action[15]]), axis=0)
                            self.pub_claw_joint.publish(claw_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        if self.qiangnao_dof_needed == 1:
                            qiangnao_msg = robotHandPosition()
                            if action[7]>0.5:
                                tem = 100
                            else:
                                tem = 0
                            qiangnao_msg.left_hand_position = list(np.concatenate(([tem], [100], [tem]*4), axis=0))
                            if action[15]>0.5:
                                tem = 100
                            else:
                                tem = 0
                            qiangnao_msg.right_hand_position = list(np.concatenate(([tem], [100], [tem]*4), axis=0))
                            self.pub_qiangnao_joint.publish(qiangnao_msg)
                        else:
                            raise KeyError("qiangnao_dof_needed != 1 is not supported!")
                elif self.which_arm == 'left':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        arm_msg = JointState()
                        arm_msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
                        arm_msg.header.stamp = rospy.Time.now()
                        arm_msg.position = np.concatenate((action[:7], self.arm_init[7:14]), axis=0)
                        self.pub_arm_joint.publish(arm_msg)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            claw_msg = lejuClawCommand()
                            claw_msg.data.position = np.concatenate(([action[7]], [0]), axis=0)
                            self.pub_claw_joint.publish(claw_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        if self.qiangnao_dof_needed == 1:
                            qiangnao_msg = robotHandPosition()
                            if action[7]>0.5:
                                tem = 100
                            else:
                                tem = 0
                            qiangnao_msg.left_hand_position = list(np.concatenate(([tem], [100], [tem]*4), axis=0))
                            qiangnao_msg.right_hand_position = [0,0,0,0,0,0]
                            self.pub_qiangnao_joint.publish(qiangnao_msg)
                        else:
                            raise KeyError("qiangnao_dof_needed != 1 is not supported!")
                elif self.which_arm == 'right':
                    if self.control_mode == 'eef':
                        raise KeyError("control_mode = 'eef' is not supported!")
                    elif self.control_mode == 'joint':
                        arm_msg = JointState()
                        arm_msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
                        arm_msg.header.stamp = rospy.Time.now()
                        arm_msg.position = np.concatenate((self.arm_init[:7],action[:7]), axis=0)
                        print("publish arm joint")
                        self.pub_arm_joint.publish(arm_msg)
                    if self.eef_type == 'leju_claw':
                        if self.leju_claw_dof_needed == 1:
                            claw_msg = lejuClawCommand()
                            claw_msg.data.position = np.concatenate(([0],[action[7]]), axis=0)
                            self.pub_claw_joint.publish(claw_msg)
                        else:
                            raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                    elif self.eef_type == 'qiangnao':
                        if self.qiangnao_dof_needed == 1:
                            qiangnao_msg = robotHandPosition()
                            qiangnao_msg.left_hand_position = [0,0,0,0,0,0]
                            if action[7]>0.5:
                                tem = 100
                            else:
                                tem = 0
                            qiangnao_msg.right_hand_position = list(np.concatenate(([tem], [100], [tem]*4), axis=0))
                            self.pub_qiangnao_joint.publish(qiangnao_msg)
                        else:
                            raise KeyError("qiangnao_dof_needed != 1 is not supported!")
                else:
                    raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")
        else:
            raise KeyError("only_arm = False is not supported!")

    def get_obs(self):
        if self.real:
            if self.only_arm:
                if self.which_arm == 'both':
                    if self.eef_type == 'qiangnao':
                        self.state = np.concatenate((self.joint_q[:7], self.qiangnao_state[:self.qiangnao_dof_needed],self.joint_q[7:],self.qiangnao_state[self.qiangnao_dof_needed:]), axis=0)
                    elif self.eef_type == 'leju_claw':
                        self.state = np.concatenate((self.joint_q[:7], self.claw_state[:self.leju_claw_dof_needed],self.joint_q[7:],self.claw_state[self.leju_claw_dof_needed:]), axis=0)
                elif self.which_arm == 'left' or self.which_arm == 'right':
                    if self.eef_type == 'qiangnao':
                        self.state = np.concatenate((self.joint_q, self.qiangnao_state), axis=0)
                    elif self.eef_type == 'leju_claw':
                        self.state = np.concatenate((self.joint_q, self.claw_state), axis=0)
                else:
                    raise KeyError("which_arm != 'left' or 'right' or 'both' is not supported!")
        obs = {"state": self.state}
        if self.only_arm:
            if self.which_arm == 'both':
                obs['images'] = {
                    "cam_h": self.cam_h_img,
                    "cam_l": self.cam_l_img,
                    "cam_r": self.cam_r_img,
                }
            elif self.which_arm == 'left':
                obs['images'] = {
                    "cam_h": self.cam_h_img,
                    "cam_l": self.cam_l_img,
                }
            elif self.which_arm == 'right':
                obs['images'] = {
                    "cam_h": self.cam_h_img,
                    "cam_r": self.cam_r_img,
                }
        return obs

    def process_rgb_img(self, msg):
        # 处理 CompressedImage
        img_arr = np.frombuffer(msg.data, dtype=np.uint8)
        # print("img_arr.max",img_arr.max())
        cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Failed to decode compressed image")
        # 色域转换由BGR->RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img=cv2.resize(cv_img,(848,480))
        img_array = np.array(cv_img)
        return img_array

    # ROS callbacks
    def cam_h_callback(self, msg):
        self.cam_h_img = self.process_rgb_img(msg)

    def cam_l_callback(self, msg):
        self.cam_l_img = self.process_rgb_img(msg)
    
    def cam_r_callback(self, msg):
        # print("cam_r_callback!")
        self.cam_r_img = self.process_rgb_img(msg)

    def F_state_callback(self, msg): # Used in simulation
        all_joint_angle = msg.position
        if self.only_arm:
            joint = all_joint_angle[:28]
            if self.which_arm == 'both':
                if self.eef_type == 'leju_claw':
                    claw = all_joint_angle[28:]
                    output_state = joint[12:19]
                    if self.leju_claw_dof_needed == 1:
                        output_state = np.insert(output_state, 7, claw[0])
                        output_state = np.concatenate((output_state, joint[19:26]), axis=0)
                        output_state = np.insert(output_state, 15, claw[8])
                    else:
                        raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
            elif self.which_arm == 'left':
                if self.eef_type == 'leju_claw':
                    claw = all_joint_angle[28:]
                    output_state = joint[12:19]
                    if self.leju_claw_dof_needed == 1:
                        output_state = np.insert(output_state, 7, claw[0])
                    else:
                        raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
            elif self.which_arm == 'right':
                if self.eef_type == 'leju_claw':
                    claw = all_joint_angle[28:]
                    output_state = joint[19:26]
                    if self.leju_claw_dof_needed == 1:
                        output_state = np.insert(output_state, 7, claw[8])
                    else:
                        raise KeyError("leju_claw_dof_needed != 1 is not supported!")
                elif self.eef_type == 'qiangnao':
                    raise KeyError("qiangnao is not supported!")
        else:
            raise KeyError("only_arm = False is not supported!")
        
        self.state = output_state

    def joint_state_callback(self, msg): # Used in real
        joint_q = msg.joint_data.joint_q
        if self.only_arm:
            self.joint_q = np.concatenate((joint_q[self.slice_robot[0][0]:self.slice_robot[0][1]], joint_q[self.slice_robot[1][0]:self.slice_robot[1][1]]), axis=0)
        else:
            raise KeyError("only_arm = False is not supported!")

    def qiangnao_state_callback(self, msg): # Used in real
        # print("qiangnao_state:",msg)
        if self.only_arm:
            # qiangnao_state = np.concatenate((msg.left_hand_position, msg.right_hand_position), axis=0)
            qiangnao_state = msg.position
            self.qiangnao_state = np.concatenate((qiangnao_state[self.qiangnao_slice[0][0]:self.qiangnao_slice[0][1]], qiangnao_state[self.qiangnao_slice[1][0]:self.qiangnao_slice[1][1]]), axis=0)
            if self.is_binary:
                self.qiangnao_state = np.where(self.qiangnao_state>50, 1, 0)
            else:
                self.qiangnao_state = self.qiangnao_state/100
        else:
            raise KeyError("only_arm = False is not supported!")

    def claw_state_callback(self, msg): # Used in real
        if self.only_arm:
            claw_state = np.concatenate((msg.data.position[self.claw_slice[0][0]:self.claw_slice[0][1]], msg.data.position[self.claw_slice[1][0]:self.claw_slice[1][1]]), axis=0)
            self.claw_state = claw_state
            if self.is_binary:
                self.claw_state = np.where(self.claw_state>50, 1, 0)
            else:
                self.claw_state = self.claw_state/100
        else:
            raise KeyError("only_arm = False is not supported!")

# Usage example
if __name__ == "__main__":
    env = KuavoBaseRosEnv()
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done:
            obs, info = env.reset()