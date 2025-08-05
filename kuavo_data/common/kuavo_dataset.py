#!/usr/bin/env python3
# pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
# pip install roslz4 --extra-index-url https://rospypi.github.io/simple/
from flask import config
import numpy as np
import cv2
import rosbag
from pprint import pprint
import os
import glob
from collections import defaultdict

# ================ 机器人关节信息定义 ================

DEFAULT_LEG_JOINT_NAMES=[
    "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
    "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll",
]
DEFAULT_ARM_JOINT_NAMES = [
    "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",
    "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",
]
DEFAULT_HEAD_JOINT_NAMES = [
    "head_yaw", "head_pitch"
]
DEFAULT_DEXHAND_JOINT_NAMES = [
    "left_qiangnao_1", "left_qiangnao_2","left_qiangnao_3","left_qiangnao_4","left_qiangnao_5","left_qiangnao_6",
    "right_qiangnao_1", "right_qiangnao_2","right_qiangnao_3","right_qiangnao_4","right_qiangnao_5","right_qiangnao_6",
]
DEFAULT_LEJUCLAW_JOINT_NAMES = [
    "left_claw", "right_claw",
]

DEFAULT_JOINT_NAMES_LIST = DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES

DEFAULT_JOINT_NAMES = {
    "full_joint_names": DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES,
    "leg_joint_names": DEFAULT_LEG_JOINT_NAMES,
    "arm_joint_names": DEFAULT_ARM_JOINT_NAMES,
    "head_joint_names": DEFAULT_HEAD_JOINT_NAMES,
}



# ================ 数据转换信息定义 ================
def init_parameters(cfg):

    global DEFAULT_CAMERA_NAMES, TRAIN_HZ, MAIN_TIMELINE_FPS, SAMPLE_DROP, CONTROL_HAND_SIDE
    global SLICE_ROBOT, SLICE_DEX, SLICE_CLAW
    global IS_BINARY, DELTA_ACTION, RELATIVE_START
    global RESIZE_W, RESIZE_H
    global ONLY_HALF_UP_BODY, USE_LEJU_CLAW, USE_QIANGNAO
    global TASK_DESCRIPTION

    
    from .config_dataset import load_config
    config = load_config(cfg)

    # 从配置文件加载基本设置
    DEFAULT_CAMERA_NAMES = config.default_camera_names
    TRAIN_HZ = config.train_hz
    MAIN_TIMELINE_FPS = config.main_timeline_fps
    SAMPLE_DROP = config.sample_drop
    CONTROL_HAND_SIDE = config.which_arm

    # 根据which_arm自动计算的切片配置
    SLICE_ROBOT = config.slice_robot
    SLICE_DEX = config.dex_slice
    SLICE_CLAW = config.claw_slice

    # 处理标志
    IS_BINARY = config.is_binary
    DELTA_ACTION = config.delta_action
    RELATIVE_START = config.relative_start

    # 图像尺寸设置
    RESIZE_W = config.resize.width
    RESIZE_H = config.resize.height

    # 根据配置自动推导的标志
    ONLY_HALF_UP_BODY = config.only_half_up_body  # 总是True当only_arm为True时
    USE_LEJU_CLAW = config.use_leju_claw  # 由eef_type决定
    USE_QIANGNAO = config.use_qiangnao  # 由eef_type决定

    TASK_DESCRIPTION = config.task_description  # 任务描述


# ================ 数据处理函数定义 ==================
class KuavoMsgProcesser:
    """
    Kuavo 话题处理函数
    """
    @staticmethod
    def process_color_image(msg):
        """
        Process the color image.
        Args:
            msg (sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage): The color image message.
        Returns:
             Dict:
                - data(np.ndarray): Image data with shape (height, width, 3).
                - "timestamp" (float): The timestamp of the image.
        """
        if hasattr(msg, 'encoding'):
            if msg.encoding != 'rgb8':
                # Handle different encodings here if necessary
                raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected 'rgb8'.")

            # Convert the ROS Image message to a numpy array
            img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

            # If the image is in 'bgr8' format, convert it to 'rgb8'
            if msg.encoding == 'bgr8':
                cv_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            else:
                cv_img = img_arr
        else:
            # 处理 CompressedImage
            img_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("Failed to decode compressed image")
            # 色域转换由BGR->RGB
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img=cv2.resize(cv_img,(RESIZE_W,RESIZE_H)) ### ATT: resize the image to 640x480(w * h)
        return {"data": cv_img, "timestamp": msg.header.stamp.to_sec()}


    @staticmethod
    def process_joint_state(msg):
        """
            Args:
                msg (kuavo_msgs/sensorsData): The joint state message.
            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        joint_q = msg.joint_data.joint_q
        return {"data": joint_q, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd(msg):
        """
            Args:
                msg (kuavo_msgs/jointCmd): The joint state message.

            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        return {"data": msg.joint_q, "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_kuavo_arm_traj(msg):
        """Process the arm trajectory command.

        Args:
            msg (sensor_msgs/JointState): The arm trajectory command message.

        Returns:
            Dict: A dictionary containing the processed arm trajectory data.
        """
        
        # radian
        return {"data": np.deg2rad(msg.position), "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_claw_state(msg):
        """
            Args:
                msg (kuavo_sdk/lejuClawState): The claw state message.
            Returns:
                Dict:
                    - data(np.ndarray): The claw state data with shape (2,).
                    - "state" (float): The state of the claws state.
        """
        state= msg.data.position
        return { "data": state, "timestamp": msg.header.stamp.to_sec() }

    @staticmethod
    def process_claw_cmd(msg):
        position= msg.data.position
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_qiangnao_state(msg):
        state= list(msg.left_hand_position)
        state.extend(list(msg.right_hand_position))
        return { "data": state, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_dex_state(msg):
        return { "data": msg.position, "timestamp": msg.header.stamp.to_sec() }

    @staticmethod
    def process_qiangnao_cmd(msg):
        position= list(msg.left_hand_position)
        position.extend(list(msg.right_hand_position))
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_sensors_data_raw_extract_imu(msg):
        imu_data = msg.imu_data
        gyro = imu_data.gyro
        acc = imu_data.acc
        free_acc = imu_data.free_acc
        quat = imu_data.quat

        # 将数据合并为一个NumPy数组
        imu = np.array([gyro.x, gyro.y, gyro.z,
                        acc.x, acc.y, acc.z,
                        free_acc.x, free_acc.y, free_acc.z,
                        quat.x, quat.y, quat.z, quat.w])

        return {"data": imu, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_sensors_data_raw_extract_arm(msg):
        """
        Processes raw joint state data from a given message by extracting the portion relevant to the arm.

        Parameters:
            msg: The input message containing joint state information.

        Returns:
            dict: A dictionary with processed joint state data. The 'data' field is sliced to include only indices 12 through 25.

        Notes:
            This function uses KuavoMsgProcesser.process_joint_state to initially process the input message and then extracts the specific range of data for further use.
        """
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_sensors_data_raw_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:]
        return res

    @staticmethod
    def process_depth_image(msg):
        """
        Process the depth image.

        Args:
            msg (sensor_msgs/Image): The depth image message.

        Returns:
            Dict:
                - data(np.ndarray): Depth image data with shape (height, width).
                - "timestamp" (float): The timestamp of the image.
        """
        # Check if the image encoding is '16UC1' which is a common encoding for depth images
        if msg.encoding != '16UC1':
            raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected '16UC1'.")

        # Convert the ROS Image message to a numpy array
        img_arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        # The depth image is already in '16UC1' format, so no conversion is needed
        return {"data": img_arr, "timestamp": msg.header.stamp.to_sec()}

class KuavoRosbagReader:
    def __init__(self):
        self._msg_processer = KuavoMsgProcesser()
        self._topic_process_map = {
            "observation.state": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_joint_state,
            },
            "action.kuavo_arm_traj": {
                "topic": "/kuavo_arm_traj",
                "msg_process_fn": self._msg_processer.process_kuavo_arm_traj,
            },
            "action": {
                "topic": "/joint_cmd",
                "msg_process_fn": self._msg_processer.process_joint_cmd,
            },
            "observation.imu": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_sensors_data_raw_extract_imu,
            },
            "observation.claw": {
                # 末端数据二指夹爪的位置状态信息
                "topic": "/leju_claw_state",
                "msg_process_fn": self._msg_processer.process_claw_state,
            },
            "action.claw": {
                # 末端数据二指夹爪的运动信息位置
                "topic": "/leju_claw_command",
                "msg_process_fn": self._msg_processer.process_claw_cmd,
            },
            "observation.qiangnao": {
                "topic": "/dexhand/state",
                "msg_process_fn": self._msg_processer.process_dex_state,
            },
            "action.qiangnao": {
                "topic": "/control_robot_hand_position",
                "msg_process_fn": self._msg_processer.process_qiangnao_cmd,
            },
        }
        for camera in DEFAULT_CAMERA_NAMES:
            # observation.images.{camera}.depth  => color images
            # if 'wrist' in camera or 'head_cam_h' in camera:
            #     self._topic_process_map[f"{camera}"] = {
            #         "topic": f"/{camera[-5:]}/color/image_raw/compressed",   # "/{camera}/color/compressed", 新刷的20.04orin镜像可以直接发布压缩图像，不用额外的压缩节点
            #         "msg_process_fn": self._msg_processer.process_color_image,
            #     }
            if 'wrist_cam_l' in camera:
                self._topic_process_map[f"{camera}"] = {
                    "topic": "/cam_l/color/image_raw/compressed",   ### ATT: 这里的cam_r是因为在2025年7月8日的rosbag中，cam_r是左手腕相机
                    "msg_process_fn": self._msg_processer.process_color_image,
                }
            elif 'wrist_cam_r' in camera:
                self._topic_process_map[f"{camera}"] = {
                    "topic": "/cam_r/color/image_raw/compressed",
                    "msg_process_fn": self._msg_processer.process_color_image,
                }
            elif 'head_cam_h' in camera:
                self._topic_process_map[f"{camera}"] = {
                    "topic": "/cam_h/color/image_raw/compressed",
                    "msg_process_fn": self._msg_processer.process_color_image,
            }
            elif 'head_cam_l' in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/zedm/zed_node/left/image_rect_color/compressed",
                "msg_process_fn": self._msg_processer.process_color_image,
            }
            elif 'head_cam_r' in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/zedm/zed_node/right/image_rect_color/compressed",
                "msg_process_fn": self._msg_processer.process_color_image,
            }

            # observation.images.{camera}.depth => depth images
            # self._topic_process_map[f"{camera}.depth"] = {
            #     "topic": f"/{camera}/depth/image_rect_raw",
            #     "msg_process_fn": self._msg_processer.process_depth_image,
            # }

    def load_raw_rosbag(self, bag_file: str):
        try:
            bag = rosbag.Bag(bag_file)      
            return bag
        except rosbag.bag.ROSBagUnindexedException:
            print(f"Bag file {bag_file} is unindexed, attempting to reindex...")
            from common.utils import reindex_rosbag
            reindexed_file = reindex_rosbag(bag_file)
            if reindexed_file:
                try:
                    bag = rosbag.Bag(reindexed_file)
                    return bag
                except Exception as e:
                    print(f"Error loading reindexed bag file: {e}")
                    raise RuntimeError(f"Failed to load reindexed bag file: {reindexed_file}")
            else:
                # 尝试允许未索引的方式打开
                print(f"Reindexing failed, trying to open with allow_unindexed=True")
                try:
                    bag = rosbag.Bag(bag_file, 'r', allow_unindexed=True)
                    print(f"Successfully opened unindexed bag: {bag_file}")
                    return bag
                except Exception as e:
                    print(f"Failed to open unindexed bag: {e}")
                    raise RuntimeError(f"Failed to reindex and load bag file: {bag_file}")
        except Exception as e:
            print(f"Error loading bag file {bag_file}: {e}")
            raise
    
    def print_bag_info(self, bag: rosbag.Bag):
        pprint(bag.get_type_and_topic_info().topics)
     
    def process_rosbag(self, bag_file: str):
        """
        Process the rosbag file and return the processed data.

        Args:
            bag_file (str): The path to the rosbag file.

        Returns:
            Dict: The processed data.
        """
        bag = self.load_raw_rosbag(bag_file)
        data = {}
        for key, topic_info in self._topic_process_map.items():
            topic = topic_info["topic"]
            msg_process_fn = topic_info["msg_process_fn"]
            data[key] = []
            for _, msg, t in bag.read_messages(topics=topic):
                msg_data = msg_process_fn(msg)
                # 如果没有 header.stamp或者时间戳是远古时间不合要求，使用bag的时间戳 
                correct_timestamp = t.to_sec() 
                msg_data["timestamp"] = correct_timestamp
                data[key].append(msg_data)
        
        data_aligned = self.align_frame_data(data)
        
        return data_aligned
    
    def align_frame_data(self, data: dict):
        aligned_data = defaultdict(list)
        main_timeline = max(
            DEFAULT_CAMERA_NAMES, 
            key=lambda cam_k: len(data.get(cam_k, []))
        )
        jump = MAIN_TIMELINE_FPS // TRAIN_HZ
        main_img_timestamps = [t['timestamp'] for t in data[main_timeline]][SAMPLE_DROP:-SAMPLE_DROP][::jump]
        min_end = min([data[k][-1]['timestamp'] for k in data.keys() if len(data[k]) > 0])
        main_img_timestamps = [t for t in main_img_timestamps if t < min_end]
        for stamp in main_img_timestamps:
            stamp_sec = stamp
            for key, v in data.items():
                if len(v) > 0:
                    this_obs_time_seq = [this_frame['timestamp'] for this_frame in v]
                    time_array = np.array([t for t in this_obs_time_seq])
                    idx = np.argmin(np.abs(time_array - stamp_sec))
                    aligned_data[key].append(v[idx])
                else:
                    aligned_data[key] = []
        print(f"Aligned {key}: {len((data[main_timeline]))} -> {len(next(iter(aligned_data.values())))}")
        for k, v in aligned_data.items():
            if len(v) > 0:
                print(v[0]['timestamp'], v[1]['timestamp'],k)
        return aligned_data
    
    
    def list_bag_files(self, bag_dir: str):
        bag_files = glob.glob(os.path.join(bag_dir, '*.bag'))
        bag_files.sort()
        return bag_files
    
    def process_rosbag_dir(self, bag_dir: str):
        all_data = []
        # 按照文件名排序，获取 bag 文件列表
        bag_files = self.list_bag_files(bag_dir)
        episode_id = 0
        for bf in bag_files:
            print(f"Processing bag file: {bf}")
            episode_data = self.process_rosbag(bf)
            all_data.append(episode_data)
        
        return all_data
    
    

if __name__ == '__main__':
    bag_file = '/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera/00001/testcamera_20250213_193331.bag'
    bag_dir = '/Users/wason/Code/RobotEmbodiedData/lerobot/data/testcamera2/'
    
    bag_file_1 = '/home/leju-ali/hx/kuavo/Task12_zed_dualArm/rosbag/rosbag_2025-03-15-15-21-40.bag'

    
    reader = KuavoRosbagReader()
    
    # reader.process_rosbag_dir(bag_dir)
    
    data_raw = reader.process_rosbag(bag_file_1)
    # data_aligned = reader.align_frame_data(data_raw)
    
    # print(data.keys())
