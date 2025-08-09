# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path

from sympy import im
from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm

from kuavo_train.wrapper.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from lerobot.utils.random_utils import set_seed
import datetime
import time
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor

def img_preprocess(image, device="cpu"):
    return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)

@hydra.main(config_path="configs", config_name="eval_config", version_base=None)
def main(cfg: DictConfig):
    use_delta = cfg.use_delta
    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    start_seed = cfg.start_seed

    # set seed
    set_seed(seed=seed)

    # Select your device
    device = torch.device(cfg.device)

    # Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
    # pretrained_policy_path = "lerobot/diffusion_pusht"
    # OR a path to a local outputs/train folder.
    timestamp = "epoch710"
    pretrained_policy_path = Path("outputs/train/example_qiao_diffusion") / timestamp

    policy = CustomDiffusionPolicyWrapper.from_pretrained(pretrained_policy_path)
    policy.to(device)
    policy.eval()

    # Create a directory to store the video of the evaluation
    output_directory = Path("outputs/eval/example_qiao_diffusion")
    output_directory = output_directory / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    # "gym_pusht/PushT-v0" or "gym_aloha/AlohaInsertion-v0"
    max_episode_steps = 150
    env = gym.make(
        "Kuavo-Real",
        max_episode_steps=max_episode_steps,
    )

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    print("policy.config.input_features",policy.config.input_features)
    print("env.observation_space",env.observation_space)

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    print("policy.config.output_features",policy.config.output_features)
    print("env.action_space",env.action_space)

    # Log evaluation results
    log_file_path = output_directory / "evaluation_log.txt"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")

    success_count = 0
    for episode in tqdm(range(eval_episodes), desc="Evaluating model", unit="episode"):
        # Reset the policy and environments to prepare for rollout
        policy.reset()
        numpy_observation, info = env.reset(seed=episode+start_seed)
        # print("numpy_observation",numpy_observation)

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []
        cam_h_frames = []
        cam_l_frames = []
        cam_r_frames = []
        cam_keys = {
            'both':  ["cam_h", "cam_l", "cam_r"],
            'left':  ["cam_h", "cam_l"],
            'right': ["cam_h", "cam_r"],
        }[env.which_arm]

        obs_keys = {
            'cam_h': "observation.images.head_cam_h",
            'cam_l': "observation.images.wrist_cam_l",
            'cam_r': "observation.images.wrist_cam_r",
        }
        frame_map = {
            "cam_h": cam_h_frames,
            "cam_l": cam_l_frames,
            "cam_r": cam_r_frames,
        }
        # Render frame of the initial state
        # frames.append(env.render())

        step = 0
        done = False
        with tqdm(total=max_episode_steps, desc=f"Episode {episode+1}", unit="step", leave=False) as pbar:
            while not done:
                start_time = time.time()
                # Prepare observation for the policy running in Pytorch
                

                # Extract camera images
                cam_images = {
                    k: img_preprocess(numpy_observation["images"][k], device=device)
                    for k in cam_keys
                }

                state = torch.from_numpy(numpy_observation["state"]).float().unsqueeze(0).to(device, non_blocking=True)

                observation = {obs_keys[k]: cam_images[k] for k in cam_keys}
                observation["observation.state"] = state


                with torch.inference_mode():
                    action = policy.select_action(observation)
                numpy_action = action.squeeze(0).cpu().numpy()
                print("numpy_action", numpy_action)

                # Clip the action to the action space limits
                if use_delta:
                    if env.real:
                        if env.which_arm == "both":
                            for i in [(0, 7), (8, 15)]:
                                numpy_action[i[0]:i[1]] = np.clip(numpy_action[i[0]:i[1]], -0.05, 0.05) + env.start_state[i[0]:i[1]]
                            env.start_state = np.concatenate([
                                np.clip(numpy_action[0:7], env.action_space.low[0:7], env.action_space.high[0:7]),
                                np.clip(numpy_action[8:15], env.action_space.low[8:15], env.action_space.high[8:15])
                            ])
                        else:
                            numpy_action[:7] = np.clip(numpy_action[:7], -0.05, 0.05) + env.start_state[:7]
                            env.start_state = np.clip(numpy_action[:7], env.action_space.low[:7], env.action_space.high[:7])
                    else:
                        numpy_action += numpy_observation["state"]

                # 执行动作
                numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
                rewards.append(reward)

                # 相机帧记录

                for k in cam_keys:
                    frame_map[k].append(cam_images[k].squeeze(0).cpu().numpy().transpose(1, 2, 0))


                # The rollout is considered done when the success state is reached (i.e. terminated is True),
                # or the maximum number of iterations is reached (i.e. truncated is True)
                done = terminated | truncated | done
                step += 1

                end_time = time.time()
                print(f"Step {step} time: {end_time - start_time:.3f}s")
                
                # Update progress bar
                status = "Success" if terminated else "Running"
                pbar.set_postfix({
                    "Reward": f"{reward:.3f}",
                    "Status": status,
                    "Total Reward": f"{sum(rewards):.3f}"
                })
                pbar.update(1)

        if terminated:
            success_count += 1
            tqdm.write(f"✅ Episode {episode+1}: Success! Total reward: {sum(rewards):.3f}")
        else:
            tqdm.write(f"❌ Episode {episode+1}: Failed! Total reward: {sum(rewards):.3f}")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.ros_rate
        
        # Encode all frames into a mp4 video.
        for cam in cam_keys:
            frames = frame_map[cam]
            output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
            imageio.mimsave(str(output_path), frames, fps=fps)

        # print(f"Video of the evaluation is available in '{video_path}'.")

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Rewards per Episode: {numpy.array(rewards).sum()}")

    with log_file_path.open("a") as log_file:
        log_file.write("\n")
        log_file.write(f"Success Count: {success_count}\n")
        log_file.write(f"Success Rate: {success_count / eval_episodes:.2f}\n")

    # Display final statistics
    print("\n" + "="*50)
    print(f"🎯 Evaluation completed!")
    print(f"📊 Success count: {success_count}/{eval_episodes}")
    print(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    print(f"📁 Videos and logs saved to: {output_directory}")
    print("="*50)
