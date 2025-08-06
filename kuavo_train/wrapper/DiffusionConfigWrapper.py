from typing import Any, Dict
from dataclasses import dataclass,field
import copy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import NormalizationMode

@dataclass
class CustomDiffusionConfigWrapper(DiffusionConfig):
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        default_map = {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }

        # 合并：传入的 YAML 会覆盖默认值，未定义的保持默认
        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)

        self.normalization_mapping = merged
