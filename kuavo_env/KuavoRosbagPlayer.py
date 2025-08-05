#!/usr/bin/env python3
import rospy
import numpy as np
import rosbag
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from env.KuavoBaseRosEnv import KuavoBaseRosEnv

class KuavoRosbagPlayer(KuavoBaseRosEnv):
    """基于KuavoBaseRosEnv的rosbag播放器"""
    
    def __init__(self, bag_file_path=None, arm_topic="/kuavo_arm_traj", eef_topic="/control_robot_hand_position"):
        super().__init__()
        self.bag_file_path = bag_file_path
        self.arm_topic = arm_topic
        self.eef_topic = eef_topic
        self.arm_bag_data = []
        self.eef_bag_data = []
        self.is_playing = False
        
        if bag_file_path:
            self.load_bag_data(bag_file_path, arm_topic, eef_topic)
    
    def load_bag_data(self, bag_file_path, arm_topic="/kuavo_arm_traj", eef_topic="/control_robot_hand_position"):
        """从rosbag中加载指定话题的数据"""
        print(f"正在从 {bag_file_path} 中读取数据...")
        print(f"手臂话题: {arm_topic}")
        print(f"末端执行器话题: {eef_topic}")
        
        try:
            with rosbag.Bag(bag_file_path, 'r') as bag:
                for topic, msg, t in bag.read_messages(topics=[arm_topic]):
                    self.arm_bag_data.append({
                        'timestamp': t.to_sec(),
                        'message': msg
                    })
                print(f"成功读取 {len(self.arm_bag_data)} 条手臂消息")
                self.arm_bag_data.sort(key=lambda x: x['timestamp'])

                for topic, msg, t in bag.read_messages(topics=[eef_topic]):
                    self.eef_bag_data.append({
                        'timestamp': t.to_sec(),
                        'message': msg
                    })
                print(f"成功读取 {len(self.eef_bag_data)} 条末端执行器消息")
                self.eef_bag_data.sort(key=lambda x: x['timestamp'])
                
        except Exception as e:
            print(f"读取rosbag失败: {e}")
            self.arm_bag_data = []
            self.eef_bag_data = []
    

    
    def reset_robot(self):
        """重置机器人到初始状态"""
        print("正在重置机器人...")
        super().reset()
        print("机器人重置完成")
    
    def play_bag_data(self, playback_speed=1.0, auto_reset=True):
        """播放rosbag中的数据"""
        if not self.arm_bag_data:
            print("没有可播放的数据")
            return
        
        # 自动重置机器人
        if auto_reset:
            self.reset_robot()
            time.sleep(2)
        
        # 播放数据
        start_playback_time = time.time()
        first_timestamp = self.arm_bag_data[0]['timestamp']
        self.is_playing = True
        
        # 初始化eef数据索引
        eef_index = 0
        arm_index = 0

        start_pos = self.joint_q.copy()/np.pi*180
        if self.which_arm == 'right':
            target_pos = self.arm_bag_data[0]['message'].position[7:]
            n_steps = 100
            msg = JointState()
            msg.name = ["arm_joint_" + str(i) for i in range(1, 15)] # Without this message, the arm will not move
            for i in range(1, n_steps + 1):
                interp_pos = start_pos + (np.array(target_pos) - start_pos) * (i / n_steps)
                msg.header.stamp = rospy.Time.now()
                msg.position = np.concatenate((self.arm_init[:7], interp_pos), axis=0) 
                # 其他字段可选填充
                self.pub_arm_joint.publish(msg)
                self.rate.sleep()

        elif self.which_arm == 'both':
            target_pos = self.arm_bag_data[0]['message'].position
            n_steps = 100
            msg = JointState()
            msg.name = ["arm_joint_" + str(i) for i in range(1, 15)] # Without this message, the arm will not move
            for i in range(1, n_steps + 1):
                interp_pos = start_pos + (np.array(target_pos) - start_pos) * (i / n_steps)
                msg.header.stamp = rospy.Time.now()
                msg.position = interp_pos
                # 其他字段可选填充
                self.pub_arm_joint.publish(msg)
                self.rate.sleep()
        
        try:
            step = 0
            while True:
                
                if rospy.is_shutdown() or not self.is_playing:
                    print("播放被中断")
                    break
                
                # 计算当前时间点
                current_time = self.arm_bag_data[0]['timestamp'] + step / self.ros_rate * playback_speed
                
                # 查找当前时间点对应的arm指令
                while (arm_index < len(self.arm_bag_data) and 
                       self.arm_bag_data[arm_index]['timestamp'] <= current_time):
                    arm_data = self.arm_bag_data[arm_index]
                    # print("arm_data.position:",arm_data['message'].position[7:])
                    self.pub_arm_joint.publish(arm_data['message'])
                    arm_index += 1

                # print("joint_q:",self.joint_q/np.pi*180)
                
                # 查找当前时间点对应的eef指令
                while (eef_index < len(self.eef_bag_data) and 
                       self.eef_bag_data[eef_index]['timestamp'] <= current_time):
                    eef_data = self.eef_bag_data[eef_index]
                    # print(eef_data)
                    self._publish_eef_command(eef_data['message'])
                    eef_index += 1
                    
                self.rate.sleep()
                step += 1

                if arm_index>=len(self.arm_bag_data) or eef_index>=len(self.eef_bag_data):
                    break
            
            print("播放完成")
            
        except KeyboardInterrupt:
            print("\n播放被用户中断")
        finally:
            self.is_playing = False
    
    def _publish_eef_command(self, eef_msg):
        """发布末端执行器命令"""
        if self.eef_type == 'leju_claw' and self.leju_claw_dof_needed == 1:
            # 夹爪控制
            if not self.real:
                claw_msg = JointState()
                claw_msg.position = eef_msg.position
                self.pub_claw_joint.publish(claw_msg)
            else:
                from kuavo_msgs.msg._lejuClawCommand import lejuClawCommand
                claw_msg = lejuClawCommand()
                claw_msg.data.position = eef_msg.data.position
                self.pub_claw_joint.publish(claw_msg)
        
        elif self.eef_type == 'qiangnao' and self.qiangnao_dof_needed == 1:
            # qiangnao控制
            if not self.real:
                return  # 仿真环境不支持qiangnao
            
            else:
                from kuavo_msgs.msg._robotHandPosition import robotHandPosition
                qiangnao_msg = robotHandPosition()
                
                # 直接使用消息中的手部位置
                qiangnao_msg.left_hand_position = eef_msg.left_hand_position
                qiangnao_msg.right_hand_position = eef_msg.right_hand_position
                
                self.pub_qiangnao_joint.publish(qiangnao_msg)
    


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从rosbag播放Kuavo机器人轨迹')
    parser.add_argument('--bag_file', default='/nvme/sfw/data/qiao/real_kuavo_333333_20250726_171544.bag', help='rosbag文件路径')
    parser.add_argument('--arm-topic', default='/kuavo_arm_traj', help='手臂话题名称')
    parser.add_argument('--eef-topic', default='/control_robot_hand_position', help='末端执行器话题名称')
    parser.add_argument('--speed', type=float, default=1.0, help='播放速度倍数')
    
    args = parser.parse_args()
    
    player = KuavoRosbagPlayer(args.bag_file, args.arm_topic, args.eef_topic)
    
    player.play_bag_data(args.speed)