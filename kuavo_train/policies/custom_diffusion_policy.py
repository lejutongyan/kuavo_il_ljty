from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
import torch

class CustomDiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
    
        def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
            """Run the batch through the model and compute the loss for training or validation."""
            batch = self.normalize_inputs(batch)

            self.config.image_features
            
            if self.config.image_features:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
            batch = self.normalize_targets(batch)
            loss = self.diffusion.compute_loss(batch)
            # no output_dict so returning None
            return loss, None