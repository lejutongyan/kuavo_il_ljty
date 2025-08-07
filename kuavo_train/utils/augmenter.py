from PIL import Image, ImageDraw
from torchvision import transforms
import random
import cv2
import numpy as np
import torch
from torch import Tensor
import torchvision
import torch.nn.functional as F
import kornia

class DeterministicAugmenterColor:
    def __init__(self):
        self.augmentations = [
            'color_jitter',
            # 'rotation',
            # 'affine',
            'none', 
            'random_mask',
            'random_border_cutout',
            'gaussian_noise',
            'gamma_correction',
            # 'perspective'
        ]
        self.params = {}
        self.aug_type = 'none'
        self.params['type'] = 'none'

    def set_random_params(self):
        self.aug_type = random.choice(self.augmentations)
        self.params['type'] = self.aug_type

        if self.aug_type == 'color_jitter':
            self.params['brightness'] = random.uniform(0.3, 0.6)
            self.params['contrast'] = random.uniform(0.3, 0.6)
            self.params['saturation'] = random.uniform(0.3, 0.6)
            self.params['hue'] = random.uniform(0, 0.1)

        elif self.aug_type == 'rotation':
            # self.params['degrees'] = random.uniform(-2.5, 2.5)
            self.params['degrees'] = -2.5

        elif self.aug_type == 'affine':
            self.params['translate'] = (random.uniform(0, 0.01), random.uniform(0, 0.01))
            self.params['scale'] = (random.uniform(0.975, 1.025))

        elif self.aug_type == 'random_mask':
            self.params['mask_size'] = (random.uniform(0.3, 0.35), random.uniform(0.3, 0.35))
            self.params['mask_position'] = (random.uniform(0, 1), random.uniform(0, 1)) 
        
        elif self.aug_type == 'random_border_cutout':
            self.params['cut_ratio'] = random.uniform(0.12, 0.18) 
            self.params['which_border'] = random.choice(['top', 'bottom', 'left', 'right'])
        
        elif self.aug_type == 'gaussian_noise':  
            self.params['mean'] = random.uniform(0, 20)  
            self.params['std'] = random.uniform(10, 40) 
        
        elif self.aug_type == 'gamma_correction':  
            self.params['gamma'] = random.uniform(0.5, 2.0) 
        
        elif self.aug_type == 'perspective':  
            self.params['max_offset'] = random.uniform(1, 3) 

    def apply_augment_sequence(self, images):
        # 保存原始形状以便恢复
        orig_shape = images.shape
        B, C, H, W = orig_shape[0], orig_shape[-3], orig_shape[-2], orig_shape[-1]
        
        # 将输入统一转换为4D张量 (B, C, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        elif images.ndim == 4:
            B, C, H, W = images.shape
            images = images.view(B, C, H, W)
        elif images.ndim == 5:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
        elif images.ndim == 6:
            B, T, N, C, H, W = images.shape
            images = images.view(B * T * N, C, H, W)
        
        # 应用增强
        if self.aug_type == 'none':
            augmented = images
        
        elif self.aug_type == 'color_jitter':
            # 生成随机参数（整个batch使用相同参数）
            brightness = 2 * self.params['brightness'] * torch.rand(1, device=images.device) + 1 - self.params['brightness']
            contrast = 2 * self.params['contrast'] * torch.rand(1, device=images.device) + 1 - self.params['contrast']
            saturation = 2 * self.params['saturation'] * torch.rand(1, device=images.device) + 1 - self.params['saturation']
            hue = 2 * self.params['hue'] * torch.rand(1, device=images.device) - self.params['hue']
            
            # 应用颜色抖动
            augmented = kornia.enhance.adjust_brightness(images, brightness)
            augmented = kornia.enhance.adjust_contrast(augmented, contrast)
            augmented = kornia.enhance.adjust_saturation(augmented, saturation)
            augmented = kornia.enhance.adjust_hue(augmented, hue)
        
        elif self.aug_type == 'random_mask':
            # 计算掩码位置和大小
            mask_h = int(self.params['mask_size'][0] * H)
            mask_w = int(self.params['mask_size'][1] * W)
            top = int(self.params['mask_position'][0] * (H - mask_h))
            left = int(self.params['mask_position'][1] * (W - mask_w))
            
            # 创建掩码
            mask = torch.ones_like(images)
            mask[:, :, top:top+mask_h, left:left+mask_w] = 0
            augmented = images * mask
        
        elif self.aug_type == 'random_border_cutout':
            cut_ratio = self.params['cut_ratio']
            cut_size_h = int(cut_ratio * H)
            cut_size_w = int(cut_ratio * W)
            border = self.params['which_border']
            
            mask = torch.ones_like(images)
            if border == 'top':
                mask[:, :, :cut_size_h, :] = 0
            elif border == 'bottom':
                mask[:, :, -cut_size_h:, :] = 0
            elif border == 'left':
                mask[:, :, :, :cut_size_w] = 0
            elif border == 'right':
                mask[:, :, :, -cut_size_w:] = 0
            augmented = images * mask
        
        elif self.aug_type == 'gaussian_noise':
            noise = torch.randn_like(images) * self.params['std'] + self.params['mean']
            augmented = images + noise
            augmented = torch.clamp(augmented, 0, 1)
        
        elif self.aug_type == 'gamma_correction':
            gamma = self.params['gamma']
            augmented = torch.pow(images, gamma)
        
        else:
            raise ValueError(f"Invalid augmentation type: {self.aug_type}")
        
        # 恢复原始形状
        augmented = augmented.view(orig_shape)
        return augmented
            
    # def apply_augment(self, image):
    #     C,H,W = image.shape
    #     assert C == 3
    #     assert image.max() < 1.01
    #     if image.shape[0] in [1,3]:
    #         # print("image.shape[0] in [1,3]")
    #         image = np.transpose(image, (1, 2, 0))
    #     if image.max()<1.01:
    #         # print("image.max < 1")
    #         image = (image * 255).astype(np.uint8)
    #     # print("AUGMENT - PRE Image shape:", image.shape, "dtype:", image.dtype)

    #     image = Image.fromarray(image)
    #     self.aug_type = self.params['type']

    #     if self.aug_type == 'color_jitter':
    #         transform = transforms.Compose([transforms.ColorJitter(
    #             brightness=self.params['brightness'],
    #             contrast=self.params['contrast'],
    #             saturation=self.params['saturation'],
    #             hue=self.params['hue']
    #         ),
    #            transforms.ToTensor(),
    #            transforms.ToPILImage() ])
    #         image = transform(image)

    #     elif self.aug_type == 'rotation':
    #         transform = transforms.Compose([transforms.RandomRotation(degrees=(self.params['degrees'], self.params['degrees'])),
    #                                         transforms.ToTensor(),
    #                                         transforms.ToPILImage()
    #                                         ])
    #         image = transform(image)

    #     elif self.aug_type == 'affine':
    #         transform = transforms.Compose([transforms.RandomAffine(
    #             degrees=0,
    #             translate=self.params['translate'],
    #             scale=(self.params['scale'], self.params['scale'])
    #         ),
    #             transforms.ToTensor(),
    #             transforms.ToPILImage()])
    #         image = transform(image)
            
    #     elif self.aug_type == 'random_mask':
    #         img_array = np.array(image)
    #         h, w, _ = img_array.shape
    #         mask_h, mask_w = int(self.params['mask_size'][0] * h), int(self.params['mask_size'][1] * w)
    #         top, left = int(self.params['mask_position'][0] * h), int(self.params['mask_position'][1] * w)

    #         bottom = min(top + mask_h, h)
    #         right = min(left + mask_w, w)
        
    #         img_array[top:bottom, left:right, :] = 0  
    #         image = img_array
            
    #     elif self.aug_type == 'random_border_cutout':
    #         img_array = np.array(image)
    #         h, w, _ = img_array.shape
    #         cut_ratio = self.params['cut_ratio']  
    #         cut_size_h = int(cut_ratio * h)
    #         cut_size_w = int(cut_ratio * w)
    #         border = self.params['which_border']
    #         if border == 'top':
    #             img_array[:cut_size_h, :, :] = 0 
    #         elif border == 'bottom':
    #             img_array[-cut_size_h:, :, :] = 0 
    #         elif border == 'left':
    #             img_array[:, :cut_size_w, :] = 0  
    #         elif border == 'right':
    #             img_array[:, -cut_size_w:, :] = 0 
    #         image = img_array
            
    #     elif self.aug_type == 'gaussian_noise': 
    #         img_array = np.array(image).astype(np.float32)  
    #         noise = np.random.normal(self.params['mean'], self.params['std'], img_array.shape) 
    #         img_array += noise  
    #         img_array = np.clip(img_array, 0, 255).astype(np.uint8) 
    #         image = img_array
            
    #     elif self.aug_type == 'gamma_correction':  
    #         img_array = np.array(image).astype(np.float32) / 255.0 
    #         gamma = self.params['gamma']
    #         img_array = np.power(img_array, gamma) 
    #         img_array = (img_array * 255).astype(np.uint8) 
    #         image = img_array
        
    #     elif self.aug_type == 'perspective':  
    #         image = np.array(image)
    #         h, w, _ = image.shape
    #         max_offset = self.params['max_offset']
    #         src_pts = np.float32([
    #             [0, 0], [w - 1, 0],
    #             [w - 1, h - 1], [0, h - 1]
    #         ])
    #         dst_pts = np.float32([
    #             [random.uniform(0, max_offset), random.uniform(0, max_offset)],
    #             [w - 1 - random.uniform(0, max_offset), random.uniform(0, max_offset)],
    #             [w - 1 - random.uniform(0, max_offset), h - 1 - random.uniform(0, max_offset)],
    #             [random.uniform(0, max_offset), h - 1 - random.uniform(0, max_offset)]
    #         ])
    #         M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    #         image = cv2.warpPerspective(image, M, (w, h))
        
    #     image = np.array(image)

    #     if image.shape[0] not in [1,3]:
    #         # print("image.shape[0] not in [1,3]")
    #         image = np.transpose(image, (2, 0, 1))
    #     if image.max()>1.01:
    #         # print("image.max > 1")
    #         image = (image / 255).astype(np.float32)

    #     assert image.shape == (C,H,W)
    #     assert image.max() < 1.01
        
    #     return image

    # def apply_augment_sequence(self, images):
    #     tem_images = images.cpu().numpy()
    #     if tem_images.ndim == 4:
    #         B, C, H, W = tem_images.shape
    #         augmented_images = []
    #         for image in tem_images:
    #             augmented_image = self.apply_augment(image)
    #             augmented_images.append(augmented_image)
    #         augmented_images = np.stack(augmented_images, axis=0)
    #         augmented_images = augmented_images.reshape(B, C, H, W)
    #     elif tem_images.ndim == 3:
    #         augmented_images = self.apply_augment(tem_images)
    #     elif tem_images.ndim == 5:
    #         B, T, C, H, W = tem_images.shape
    #         tem_images = tem_images.reshape(B * T, C, H, W)
    #         augmented_images = []
    #         for image in tem_images:
    #             augmented_image = self.apply_augment(image)
    #             augmented_images.append(augmented_image)
    #         augmented_images = np.stack(augmented_images, axis=0)
    #         augmented_images = augmented_images.reshape(B, T, C, H, W)
    #     elif tem_images.ndim == 6:
    #         B, T, N, C, H, W = tem_images.shape
    #         tem_images = tem_images.reshape(B * T * N, C, H, W)
    #         augmented_images = []
    #         for image in tem_images:
    #             augmented_image = self.apply_augment(image)
    #             augmented_images.append(augmented_image)
    #         augmented_images = np.stack(augmented_images, axis=0)
    #         augmented_images = augmented_images.reshape(B, T, N, C, H, W)
    #     return torch.tensor(augmented_images,dtype=images.dtype,device=images.device)


class DeterministicAugmenterGeo4Rgbds:
    def __init__(self):
        self.augmentations = [
            'none', 
            'rotation',
            'translation'
        ]
        self.params = {}
        self.aug_type = 'none'
        self.params['type'] = 'none'

    def set_random_params(self):
        self.aug_type = random.choice(self.augmentations)
        self.params['type'] = self.aug_type

        if self.aug_type == 'rotation':
            self.params['degrees'] = random.uniform(-5, 5)
        
        elif self.aug_type == 'translation':
            self.params['shift_x'] = random.uniform(-8, 8) 
            self.params['shift_y'] = random.uniform(-8, 8) 
            
    def apply_augment(self, image):
        if image.shape[-1] == 5:
            transposed = True
            image = np.transpose(image, (2,0,1))
        else:
            transposed = False
        
        assert image.shape[0] == 5
        
        self.aug_type = self.params['type']

        if self.aug_type == 'rotation':
            (h, w) = image.shape[1:] 
            M = cv2.getRotationMatrix2D((w//2, h//2), self.params['degrees'], 1.0)
            rotated = np.zeros_like(image)
            rotated[0] = cv2.warpAffine(image[0], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[1] = cv2.warpAffine(image[1], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[2] = cv2.warpAffine(image[2], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[3] = cv2.warpAffine(image[3], M, (w, h), flags=cv2.INTER_NEAREST)  
            rotated[4] = cv2.warpAffine(image[4], M, (w, h), flags=cv2.INTER_NEAREST)  
            image = rotated
            
        elif self.aug_type == 'translation':
            (h, w) = image.shape[1:]
            M = np.float32([[1, 0, self.params['shift_x']], [0, 1, self.params['shift_y']]])
            translated = np.zeros_like(image)
            translated[0] = cv2.warpAffine(image[0], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) 
            translated[1] = cv2.warpAffine(image[1], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            translated[2] = cv2.warpAffine(image[2], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            translated[3] = cv2.warpAffine(image[3], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)  
            translated[4] = cv2.warpAffine(image[4], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) 
            image = translated
        
        if transposed:
            image = np.transpose(image, (1,2,0))
        
        return np.array(image)

    def apply_augment_sequence(self, images):
        if images.ndim == 4:
            augmented_images = []
            for image in images:
                augmented_image = self.apply_augment(image)
                augmented_images.append(augmented_image)
        elif images.ndim == 3:
            augmented_images = self.apply_augment(images)
        return np.array(augmented_images)
    
class DeterministicAugmenterGeo4Rgbdss:
    def __init__(self):
        self.augmentations = [
            'none', 
            'rotation',
            'translation'
        ]
        self.params = {}
        self.aug_type = 'none'
        self.params['type'] = 'none'

    def set_random_params(self):
        self.aug_type = random.choice(self.augmentations)
        self.params['type'] = self.aug_type

        if self.aug_type == 'rotation':
            self.params['degrees'] = random.uniform(-5, 5)
        
        elif self.aug_type == 'translation':
            self.params['shift_x'] = random.uniform(-8, 8) 
            self.params['shift_y'] = random.uniform(-8, 8) 
            
    def apply_augment(self, image):
        if image.shape[-1] == 6:
            transposed = True
            image = np.transpose(image, (2,0,1))
        else:
            transposed = False
        
        assert image.shape[0] == 6
        
        self.aug_type = self.params['type']

        if self.aug_type == 'rotation':
            (h, w) = image.shape[1:] 
            M = cv2.getRotationMatrix2D((w//2, h//2), self.params['degrees'], 1.0)
            rotated = np.zeros_like(image)
            rotated[0] = cv2.warpAffine(image[0], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[1] = cv2.warpAffine(image[1], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[2] = cv2.warpAffine(image[2], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[3] = cv2.warpAffine(image[3], M, (w, h), flags=cv2.INTER_NEAREST)  
            rotated[4] = cv2.warpAffine(image[4], M, (w, h), flags=cv2.INTER_NEAREST)  
            rotated[5] = cv2.warpAffine(image[5], M, (w, h), flags=cv2.INTER_NEAREST)  
            image = rotated
            
        elif self.aug_type == 'translation':
            (h, w) = image.shape[1:]
            M = np.float32([[1, 0, self.params['shift_x']], [0, 1, self.params['shift_y']]])
            translated = np.zeros_like(image)
            translated[0] = cv2.warpAffine(image[0], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) 
            translated[1] = cv2.warpAffine(image[1], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            translated[2] = cv2.warpAffine(image[2], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            translated[3] = cv2.warpAffine(image[3], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)  
            translated[4] = cv2.warpAffine(image[4], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) 
            translated[5] = cv2.warpAffine(image[5], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) 
            image = translated
        
        if transposed:
            image = np.transpose(image, (1,2,0))
        
        return np.array(image)

    def apply_augment_sequence(self, images):
        if images.ndim == 4:
            augmented_images = []
            for image in images:
                augmented_image = self.apply_augment(image)
                augmented_images.append(augmented_image)
        elif images.ndim == 3:
            augmented_images = self.apply_augment(images)
        return np.array(augmented_images)
    
class DeterministicAugmenterGeo4Rgbd:
    def __init__(self):
        self.augmentations = [
            'none', 
            'rotation',
            'translation'
        ]
        self.params = {}
        self.aug_type = 'none'
        self.params['type'] = 'none'

    def set_random_params(self):
        self.aug_type = random.choice(self.augmentations)
        self.params['type'] = self.aug_type

        if self.aug_type == 'rotation':
            self.params['degrees'] = random.uniform(-5, 5)
        
        elif self.aug_type == 'translation':
            self.params['shift_x'] = random.uniform(-8, 8) 
            self.params['shift_y'] = random.uniform(-8, 8) 
            
    def apply_augment(self, image):
        if image.shape[-1] == 4:
            transposed = True
            image = np.transpose(image, (2,0,1))
        else:
            transposed = False
        
        assert image.shape[0] == 4
        
        self.aug_type = self.params['type']

        if self.aug_type == 'rotation':
            (h, w) = image.shape[1:] 
            M = cv2.getRotationMatrix2D((w//2, h//2), self.params['degrees'], 1.0)
            rotated = np.zeros_like(image)
            rotated[0] = cv2.warpAffine(image[0], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[1] = cv2.warpAffine(image[1], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[2] = cv2.warpAffine(image[2], M, (w, h), flags=cv2.INTER_LINEAR)  
            rotated[3] = cv2.warpAffine(image[3], M, (w, h), flags=cv2.INTER_NEAREST)  
            image = rotated
            
        elif self.aug_type == 'translation':
            (h, w) = image.shape[1:]
            M = np.float32([[1, 0, self.params['shift_x']], [0, 1, self.params['shift_y']]])
            translated = np.zeros_like(image)
            translated[0] = cv2.warpAffine(image[0], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) 
            translated[1] = cv2.warpAffine(image[1], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            translated[2] = cv2.warpAffine(image[2], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            translated[3] = cv2.warpAffine(image[3], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)  
            image = translated
        
        if transposed:
            image = np.transpose(image, (1,2,0))
        
        return np.array(image)

    def apply_augment_sequence(self, images):
        if images.ndim == 4:
            augmented_images = []
            for image in images:
                augmented_image = self.apply_augment(image)
                augmented_images.append(augmented_image)
        elif images.ndim == 3:
            augmented_images = self.apply_augment(images)
        return np.array(augmented_images)


class NoiseAdder_AfterNorm:
    def __init__(self, state_noise_scale=0.002, action_noise_scale=0.001):
        self.state_noise_scale = state_noise_scale
        self.action_noise_scale = action_noise_scale

    def add_noise(self, 
                  nsample, 
                  keys=[
                        'joint_angle', 
                        'joint_vel', 
                        'joint_cur', 
                        'eef_pose', 
                        # 'joint_actions', 
                        'eef_actions'
                        ]
                  ):
        noisy_nsample = nsample.copy()

        for key in keys:
            noisy_nsample[key] += np.random.normal(0, 
                                                   self.state_noise_scale if 'actions' not in key else self.action_noise_scale, 
                                                   noisy_nsample[key].shape)
        return noisy_nsample
    
class NoiseAdder_AfterNorm2:
    def __init__(self, state_noise_scale=0.002, action_noise_scale=0.001):
        self.state_noise_scale = state_noise_scale
        self.action_noise_scale = action_noise_scale

    def add_noise(self, 
                  nsample, 
                  keys=[
                        'joint_angle', 
                        'joint_vel', 
                        'joint_cur', 
                        'eef_pose', 
                        # 'joint_actions', 
                        'eef_actions'
                        ]
                  ):
        
        if np.random.rand() < 0.2:
            return nsample

        noisy_nsample = nsample.copy()

        for key in keys:
            scale = self.state_noise_scale if 'actions' not in key else self.action_noise_scale
            noise = np.random.normal(0, scale, noisy_nsample[key].shape)
            noisy_nsample[key] += noise

        return noisy_nsample
    
def resize_image(image, target_size, image_type='rgb'):
    """
    Resizes the input image to the target size.
    
    Parameters:
    - image: np.ndarray or list of np.ndarray, input image data, can be of shape 
             (x, y, c), (c, x, y), (batch_size, c, x, y), (batch_size, x, y, c).
             If a list is provided, each element will be resized separately.
    - target_size: tuple (width, height), target size for resizing.
    - image_type: str, 'rgb' or 'depth', determines the interpolation method.
    
    Returns:
    - resized_image: np.ndarray or list of np.ndarray, resized image(s) with the same input format.
    """
    assert image_type in ['rgb', 'depth'], "image_type must be 'rgb' or 'depth'"

    # Determine interpolation method
    interpolation = cv2.INTER_LINEAR if image_type == 'rgb' else cv2.INTER_NEAREST
    new_H, new_W = target_size

    if isinstance(image, Tensor):
        tem_image = image.cpu().numpy()
        if tem_image.ndim == 3:
            # (x, y, c) or (c, x, y)
            if tem_image.shape[0] in [1, 3]:  # 认为是 (c, x, y)
                tem_image = np.transpose(tem_image, (1, 2, 0))
                resized_image = cv2.resize(tem_image, target_size, interpolation=interpolation)
                resized_image = np.transpose(resized_image, (2, 0, 1))
            else:  # 认为是 (x, y, c)
                resized_image = cv2.resize(tem_image, target_size, interpolation=interpolation)

        elif tem_image.ndim == 4:
            # (batch_size, c, x, y) or (batch_size, x, y, c)
            if tem_image.shape[1] in [1, 3]:  # 认为是 (batch_size, c, x, y)
                resized_images = []
                for img in tem_image:
                    img = np.transpose(img, (1, 2, 0))
                    resized_img = cv2.resize(img, target_size, interpolation=interpolation)
                    resized_images.append(np.transpose(resized_img, (2, 0, 1)))
                resized_image = np.stack(resized_images, axis=0)
            else:  # 认为是 (batch_size, x, y, c)
                resized_images = [cv2.resize(img, target_size, interpolation=interpolation) for img in tem_image]
                resized_image = np.stack(resized_images, axis=0)
        elif tem_image.ndim == 5:
            if tem_image.shape[2] in [1, 3]:
                B, T, C, H, W = tem_image.shape
                tem_image = tem_image.reshape(B * T, C, H, W)
                resized_images = []
                for img in tem_image:
                    img = np.transpose(img, (1, 2, 0))
                    resized_img = cv2.resize(img, target_size, interpolation=interpolation)
                    resized_images.append(np.transpose(resized_img, (2, 0, 1)))
                resized_image = np.stack(resized_images, axis=0)
                resized_image = resized_image.reshape(B, T, C, new_H, new_W)
            else:
                B, T, H, W, C = tem_image.shape
                tem_image = tem_image.reshape(B * T, H, W, C)
                resized_images = [cv2.resize(img, target_size, interpolation=interpolation) for img in tem_image]
                resized_image = np.stack(resized_images, axis=0)
                resized_image = resized_image.reshape(B, T, new_H, new_W, C)
        elif tem_image.ndim == 6:
            if tem_image.shape[3] in [1, 3]:
                B, T, N, C, H, W = tem_image.shape
                tem_image = tem_image.reshape(B * T * N, C, H, W)
                resized_images = []
                for img in tem_image:
                    img = np.transpose(img, (1, 2, 0))
                    resized_img = cv2.resize(img, target_size, interpolation=interpolation)
                    resized_images.append(np.transpose(resized_img, (2, 0, 1)))
                resized_image = np.stack(resized_images, axis=0)
                resized_image = resized_image.reshape(B, T, N, C, new_H, new_W)
            else:
                B, T, N, H, W, C = tem_image.shape
                tem_image = tem_image.reshape(B * T * N, H, W, C)
                resized_images = [cv2.resize(img, target_size, interpolation=interpolation) for img in tem_image]
                resized_image = np.stack(resized_images, axis=0)
                resized_image = resized_image.reshape(B, T, N, new_H, new_W, C)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return torch.tensor(resized_image,dtype=image.dtype,device=image.device)

    raise TypeError("Input must be a numpy array or a list of numpy arrays")

def crop_image(image, target_range, random_crop=False):
    """
    Crops the input image to the specified target range.
    
    Parameters:
    - image: np.ndarray or list of np.ndarray, input image data, which can have one of the following shapes:
             (x, y, c), (c, x, y), (batch_size, x, y, c), or (batch_size, c, x, y).
             If a list is provided, each element will be cropped separately.
    - target_range: list or tuple with two tuples: [(x_start, x_end), (y_start, y_end)]
                    defining the crop region.
    - image_type: str, 'rgb' or 'depth' (保留此参数以保持接口一致，但裁剪过程中不使用)
    
    Returns:
    - cropped_image: np.ndarray or list of np.ndarray, the cropped image(s) with the same format as the input.
    """
    # 确认 target_range 格式正确
    try:
        target_range = list(target_range)
    except Exception as e:
        try:
            target_range = tuple(target_range)
        except Exception:
            raise ValueError("target_range must be convertible to a list or tuple") from e
    assert isinstance(target_range, (list, tuple))

    if isinstance(image, Tensor):
        if isinstance(target_range[0],(list,tuple)) and not random_crop:
            (x_start, x_end), (y_start, y_end) = target_range
            tem_image = image.cpu().numpy()
            if tem_image.ndim == 3:
                # 3 维图像
                if tem_image.shape[0] in [1, 3, 4, 5]:
                    # 认为图像格式为 (c, x, y)
                    cropped_image = tem_image[:, x_start:x_end, y_start:y_end]
                else:
                    # 认为图像格式为 (x, y, c)
                    cropped_image = tem_image[x_start:x_end, y_start:y_end, ...]
            elif tem_image.ndim == 4:
                # 4 维图像（批量图像）
                if tem_image.shape[1] in [1, 3, 4, 5]:
                    # 认为图像格式为 (batch_size, c, x, y)
                    cropped_image = tem_image[:, :, x_start:x_end, y_start:y_end]
                else:
                    # 认为图像格式为 (batch_size, x, y, c)
                    cropped_image = tem_image[:, x_start:x_end, y_start:y_end, ...]
            elif tem_image.ndim == 5:
                if tem_image.shape[2] in [1, 3, 4, 5]:
                    cropped_image = tem_image[:, :, :, x_start:x_end, y_start:y_end]
                else:
                    cropped_image = tem_image[:, :, x_start:x_end, y_start:y_end, ...]
            elif tem_image.ndim == 6:
                if tem_image.shape[3] in [1, 3, 4, 5]:
                    cropped_image = tem_image[:, :, :, :, x_start:x_end, y_start:y_end]
                else:
                    cropped_image = tem_image[:, :, :, x_start:x_end, y_start:y_end, ...]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

            return torch.tensor(cropped_image,dtype=image.dtype,device=image.device)
        else:
            if random_crop:
                crop = torchvision.transforms.RandomCrop(target_range)
            else:
                crop = torchvision.transforms.CenterCrop(target_range)
            return crop(image)

    raise TypeError("Input must be a numpy array or a list of numpy arrays")



class Augmenter():
    def __init__(self,config):
        if config.RGB_Augmentation:
            self.RGB_Augmenter = DeterministicAugmenterColor()
        else:
            self.RGB_Augmenter = None

        self.crop_shape = config.crop_shape

        self.resize_shape = config.resize_shape