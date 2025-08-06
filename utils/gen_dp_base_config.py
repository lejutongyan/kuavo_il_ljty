from kuavo_train.wrapper.DiffusionConfigWrapper import CustomDiffusionConfigWrapper
from omegaconf import OmegaConf
from dataclasses import asdict
from collections import OrderedDict
import os

cfg = CustomDiffusionConfigWrapper()

# 用 OrderedDict 确保构造顺序
ordered_cfg = OrderedDict()
ordered_cfg["_target_"] = "wrapper.DiffusionConfigWrapper.CustomDiffusionConfigWrapper"
ordered_cfg.update(asdict(cfg))

# 转成普通 dict 保留顺序
cfg_dict = dict(ordered_cfg)

# 转成 OmegaConf
cfg_omega = OmegaConf.create(cfg_dict)

# 确保目录存在
save_path = "kuavo_train/configs/base_diffusion_config.yaml"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 保存
OmegaConf.save(cfg_omega, save_path)
print(f"配置已保存到: {save_path}")
