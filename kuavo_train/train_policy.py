import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore


@hydra.main(config_path="configs", config_name="custom_dp_config", version_base=None)
def main(cfg: DictConfig):

    root = cfg.training_params.root
    # Create a output_directory.
    output_directory = Path(cfg.training_params.output_directory)
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
    dp_cfg = instantiate(cfg.diffusion_configs, input_features=input_features, output_features=output_features, device=device)

    print("rgb_image_features",dp_cfg.rgb_image_features)  # 目前数转里头没有FeatureType.RGB和FeatureType.DEPTH，所以会报错。
    # dp_cfg.validate_features()  # 目前数转里头没有FeatureType.RGB和FeatureType.DEPTH，所以会报错。
    print("Vision backbone:", dp_cfg.vision_backbone)
    print("Normalization mapping:", dp_cfg.normalization_mapping)
    print("Use UNet:", dp_cfg.custom.use_unet)
    print("optimizer_lr:", dp_cfg.optimizer_lr)
    # dp_cfg = DiffusionConfig(input_features=input_features, output_features=output_features, device=device)


    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(dp_cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

if __name__ == "__main__":
    main()