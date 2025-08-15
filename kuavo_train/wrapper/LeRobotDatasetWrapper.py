from lerobot.datasets.lerobot_dataset import LeRobotDataset
from collections.abc import Callable
from pathlib import Path

from pandas import notna
import torchvision
import torchvision.transforms.functional

class CustomLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
        super().__init__(repo_id,
                         root,
                         episodes,
                         image_transforms,
                         delta_timestamps,
                         tolerance_s,
                         revision,
                         force_cache_sync,
                         download_videos,
                         video_backend,
                         batch_encoding_size,
                         )
    

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            depth_keys = [key for key in item.keys() if "depth" in key]
            if len(depth_keys)==0:
                for cam in image_keys:
                    item[cam], _no, __no_use = self.image_transforms(item[cam])
            else:
                for rgb_cam, depth_cam in zip(image_keys,depth_keys):
                    # print(item[rgb_cam].dtype,item[depth_cam].dtype,item[rgb_cam].shape,item[depth_cam].shape,item[rgb_cam].max(),item[depth_cam].max())
                    item[rgb_cam], crop_position, resize_shape = self.image_transforms(item[rgb_cam])
                    if len(crop_position) == 4:
                        item[depth_cam] = torchvision.transforms.functional.crop(item[depth_cam],*crop_position)
                    else:
                        item[depth_cam] = torchvision.transforms.functional.center_crop(item[depth_cam],crop_position)
                    if resize_shape is not None:
                        item[depth_cam] = torchvision.transforms.functional.resize(item[depth_cam], resize_shape)
                    # print(item[rgb_cam].dtype,item[depth_cam].dtype,item[rgb_cam].shape,item[depth_cam].shape,item[rgb_cam].max(),item[depth_cam].max())



        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item