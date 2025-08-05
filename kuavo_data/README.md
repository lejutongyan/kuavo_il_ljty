# Kuavo数据转换工具

本工具用于将Kuavo机器人的ROS数据包转换为LeRobot数据集格式。主要包含数据收集和数据转换两个部分。

## 数据采集注意事项
1. 强脑灵巧手有6个自由度。标准握拳状态[100] * 6, 张开状态[0] * 6。不需要精细操作或者多指协同操作时，通常为设置为1，表示只需要第一个关节作为开合依据,此时需要用[0, 100, 0, 0, 0, 0]表示张开状态, [100] * 6表示握拳状态，为达到此效果，需要在使用手柄数据采集时，用拇指触碰X,B让手在打开时拇指垂直于掌心平面。


## 数据采集方法
1. 建议使用`kuavo_data_pilot`的数据采集软件,提供了优秀的ui交互和错误bag检测。
2. 也可以自己使用`rosbag record`
3. 也可以使用这里的`record.py`，需根据实际情况修改

### 话题列表

默认记录以下ROS话题(以灵巧手为例)
- `/cam_h/color/image_raw/compressed`: 右手腕相机压缩图像 (30Hz)
- `/cam_l/color/image_raw/compressed`: 左手腕相机压缩图像 (30Hz)
- `/cam_r/color/image_raw/compressed`: 右手腕相机压缩图像 (30Hz)
- `/joint_cmd`: 关节控制命令 (500Hz)
- `/sensors_data_raw`: 传感器原始数据 (500Hz)
- `/control_robot_hand_position`: 手部位置控制 (100Hz)
- `/control_robot_hand_position_state`: 手部位置状态 (100Hz)

### record.py使用方法

```bash
python collect_data/record.py [-h] [-b BAG_FOLDER_PATH] [-c CNT] [-d DURATION] [-w WAIT]

可选参数:
  -h, --help            显示帮助信息
  -b BAG_FOLDER_PATH    存储rosbag数据的目录，默认为当前目录
  -c CNT                录制次数（默认25次）
  -d DURATION           每次录制的持续时间（秒），默认20s
  -w WAIT              每次录制间的等待时间（秒），默认5s
```

### 数据检查
录制前会自动进行以下检查：
1. 话题频率检查：确保各个话题的发布频率在预期范围内
2. 时间戳检查：确保所有话题都包含时间戳信息

## 数据转换配置

### 配置文件
使用`lerobot_dataset.yaml`进行配置：

```yaml
only_arm: true
eef_type: dex_hand  # 'dex_hand' 或 'leju_claw'
which_arm: right    # 'left', 'right' 或 'both'

train_hz: 10
main_timeline: wrist_cam_h
main_timeline_fps: 30
sample_drop: 10

dex_dof_needed: 1   # 通常为1，表示只需要第一个关节作为开合依据

is_binary: false
delta_action: false
relative_start: false

resize:
  width: 640
  height: 480
```

### 主要配置项说明
- `eef_type`: 末端执行器类型，可选'dex_hand'（灵巧手）或'leju_claw'（乐聚夹爪）
- `which_arm`: 使用的手臂，可选'left'（左臂）、'right'（右臂）或'both'（双臂）
- `train_hz`: 训练数据的采样频率
- `main_timeline`: 用于时间对齐的主时间线话题
- `dex_dof_needed`: 灵巧手使用的自由度数量

## 数据转换

### 使用方法

```bash
python cvt_rosbag2lerobot.py --raw_dir RAW_DIR [-n NUM_OF_BAG] [-v PROCESS_VERSION]

必选参数:
  --raw_dir RAW_DIR    原始ROS包目录路径

可选参数:
  -n NUM_OF_BAG       要处理的包文件数量
  -v PROCESS_VERSION  处理版本（默认'v0'）
```

### 转换过程
1. 读取rosbag数据
2. 根据配置进行数据预处理：
   - 可选的二值化处理
   - 可选的相对起点处理
   - 可选的增量动作处理
3. 根据配置选择相应的关节数据：
   - 支持仅上半身或全身数据
   - 根据末端执行器类型选择相应的数据
4. 进行时间对齐
5. 保存为LeRobot数据集格式

### 输出数据
转换后的数据将保存在`{raw_dir}/../{version}/lerobot/`目录下，包含：
- 状态数据（关节角度等）
- 动作数据（关节命令等）
- 图像数据（相机图像）
- 配置信息（关节名称、数据结构等）
