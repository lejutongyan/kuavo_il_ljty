
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from kuavo_train.wrapper.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import instantiate

from diffusers.optimization import get_scheduler
# from diffusers.training_utils import EMAModel
from tqdm import tqdm

from omegaconf import DictConfig




@hydra.main(config_path="configs", config_name="custom_dp_config", version_base=None)
def main(cfg: DictConfig):

    root = cfg.training_params.root
    # Create a output_directory.
    output_directory = Path(cfg.training_params.output_directory)/ cfg.method
    timestamp = cfg.training_params.timestamp
    output_directory = output_directory / f"run_{timestamp}"

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=str(output_directory))

    # # Select your device
    device = torch.device(cfg.training_params.device)


    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_epoch = cfg.training_params.training_epoch
    batch_size = cfg.training_params.batch_size
    save_freq_epoch = cfg.training_params.save_freq_epoch
    log_freq = cfg.training_params.log_freq

    dataset_metadata = LeRobotDatasetMetadata(cfg.training_params.repoid, root=root)
    total_frames = dataset_metadata.info['total_frames']
    features = dataset_to_policy_features(dataset_metadata.features)
    print("Original dataset features:", dataset_metadata.features)
    print(f"Dataset features: {features}")

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"Input features: {input_features}")
    print(f"Output features: {output_features}")

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    dp_cfg = instantiate(cfg.diffusion_configs, input_features=input_features, output_features=output_features, device=cfg.training_params.device)

    print("rgb_image_features",dp_cfg.rgb_image_features)  # 目前数转里头没有FeatureType.RGB和FeatureType.DEPTH，所以会报错。
    # dp_cfg.validate_features()  # 目前数转里头没有FeatureType.RGB和FeatureType.DEPTH，所以会报错。
    print("Vision backbone:", dp_cfg.vision_backbone)
    print("Normalization mapping:", dp_cfg.normalization_mapping)
    print("Use UNet:", dp_cfg.use_unet)
    print("optimizer_lr:", dp_cfg.optimizer_lr)
    # dp_cfg = DiffusionConfig(input_features=input_features, output_features=output_features, device=device)
    # We can now instantiate our policy with this config and the dataset stats.
    # print("dataset_stats:", dataset_metadata.stats)
    

    policy = CustomDiffusionPolicyWrapper(dp_cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # Extract keys containing "observation" and "action" from metadata.info['features']
    delta_timestamps_keys = {
        key: value for key, value in dataset_metadata.info['features'].items()
        if "observation" in key or "action" in key
    }
    delta_timestamps = {}
    for key in delta_timestamps_keys:
        if "observation" in key:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in dp_cfg.observation_delta_indices]
        elif "action" in key:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in dp_cfg.action_delta_indices]


    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    # delta_timestamps = {
    #     "observation.images.top": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    # }

    # delta_timestamps = {
    #     "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    # }

    # # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    # delta_timestamps = {
    #     # Load the previous image and state at -0.1 seconds before current frame,
    #     # then load current image and state corresponding to 0.0 second.
    #     "observation.image": [-0.1, 0.0],
    #     "observation.state": [-0.1, 0.0],
    #     # Load the previous action (-0.1), the next action to be executed (0.0),
    #     # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    #     # used to supervise the policy.
    #     "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    # }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(cfg.training_params.repoid, delta_timestamps=delta_timestamps,root=root)

    # EMA and optimizer
    # ema = EMAModel(model=policy, parameters=policy.parameters(), power=cfg.training_params.ema_power)

    optimizer = torch.optim.AdamW(params=policy.parameters(), lr=cfg.training_params.learning_rate, weight_decay=cfg.training_params.weight_decay)
    
    updates_per_epoch = (total_frames // (batch_size * cfg.training_params.accumulation_steps)) + 1
    total_update_steps = training_epoch * updates_per_epoch
    
    lr_scheduler = get_scheduler(name=cfg.training_params.scheduler_name, 
                                 optimizer=optimizer, 
                                 num_warmup_steps=cfg.training_params.scheduler_warmup_steps, 
                                 num_training_steps=total_update_steps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.training_params.num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=cfg.training_params.drop_last,
    )

    # Run training loop.
    step = 0
    for epoch in range(training_epoch):
        # 使用 tqdm 进度条（推荐，直观）
        epoch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{training_epoch}")
        for batch_idx, batch in epoch_bar:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss /= cfg.training_params.accumulation_steps
            loss.backward()
            if batch_idx % cfg.training_params.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # ema.step(policy)
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], step)

            if step % log_freq == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                epoch_bar.set_postfix({"loss": f"{(loss * cfg.training_params.accumulation_steps).item():.3f}", "step": step})
                # 传统文本输出方式（如需切换，取消注释下行并注释上行 set_postfix）
                # print(f"epoch: {epoch+1}/{training_epoch} | batch: {batch_idx+1}/{len(dataloader)} | step: {step} | loss: {loss.item():.3f}")
            if (epoch+1) % save_freq_epoch == 0:
                tem_directory = output_directory / f"epoch{epoch}"
                policy.save_pretrained(tem_directory)
            step += 1

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)
    writer.close()

if __name__ == "__main__":
    main()