from typing import Any, Dict
from dataclasses import dataclass,fields,field
import copy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from omegaconf import DictConfig, OmegaConf, ListConfig

@DiffusionConfig.register_subclass("custom_diffusion")
@dataclass
class CustomDiffusionConfigWrapper(DiffusionConfig):
    custom: Dict[str, Any] = field(default_factory=dict)
    use_custom_override: bool = False

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

        if isinstance(self.custom, DictConfig):
            for k, v in self.custom.items():
                if self.use_custom_override or not hasattr(self, k):
                    setattr(self, k, v)
        self.input_features = self._normalize_feature_dict(self.input_features)
        self.output_features = self._normalize_feature_dict(self.output_features)
        self._convert_omegaconf_fields()

    def _normalize_feature_dict(self, d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }
    
    def _convert_omegaconf_fields(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (ListConfig, DictConfig)):
                converted = OmegaConf.to_container(val, resolve=True)
                setattr(self, f.name, converted)

    @property
    def rgb_image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.RGB}
    
    @property
    def depth_image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.DEPTH}

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.rgb_image_features==0 and self.depth_image_features==0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            if isinstance(self.crop_shape[0],(list,tuple)):
                (x_start, x_end), (y_start, y_end) = self.crop_shape
                for key, image_ft in self.image_features.items():
                    if x_start < 0 or x_end > image_ft.shape[1] or y_start<0 or y_end > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                            f"for `crop_shape` and {image_ft.shape} for "
                            f"`{key}`."
                        )
                for key, image_ft in self.rgb_image_features.items():
                    if x_start < 0 or x_end > image_ft.shape[1] or y_start<0 or y_end > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                            f"for `crop_shape` and {image_ft.shape} for "
                            f"`{key}`."
                        )
                for key, image_ft in self.depth_image_features.items():
                    if x_start < 0 or x_end > image_ft.shape[1] or y_start<0 or y_end > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                            f"for `crop_shape` and {image_ft.shape} for "
                            f"`{key}`."
                        )
            else:
                for key, image_ft in self.image_features.items():
                    if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                            f"for `crop_shape` and {image_ft.shape} for "
                            f"`{key}`."
                        )

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.rgb_image_features.items()))
        for key, image_ft in self.rgb_image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )
        for key, image_ft in self.depth_image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

