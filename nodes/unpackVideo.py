"""
UnpackVideo 节点 - 将VideoData解为ComfyUI兼容的IMAGE张量
"""

import os
import tempfile
from PIL import Image, ImageOps
import torch
import numpy as np
import comfy
from ..video_types import VideoData
from ..func import extract_frames_from_bytes


class UnpackVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "fps", "width", "height")
    FUNCTION = "unpack_video"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def unpack_video(self, video, scale_factor=1.0):
        try:
            # 确保输入是VideoData对象
            if not isinstance(video, VideoData):
                raise ValueError("Input must be a VideoData object")

            # 创建临时目录存放提取的帧
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 从视频字节中提取帧
                total_frames, fps, width, height = extract_frames_from_bytes(
                    video.to_bytes(),
                    tmp_dir,
                    fps_scale=scale_factor if scale_factor != 1.0 else None
                )

                # 加载提取的帧为IMAGE张量
                images = []
                frame_files = sorted([f for f in os.listdir(tmp_dir) if f.startswith('frame_')])

                for frame_file in frame_files:
                    frame_path = os.path.join(tmp_dir, frame_file)
                    try:
                        with Image.open(frame_path) as img:
                            img = ImageOps.exif_transpose(img).convert("RGB")
                            # 转换为张量 (1, height, width, channels)
                            image_tensor = torch.from_numpy(
                                np.array(img).astype(np.float32) / 255.0
                            ).unsqueeze(0)
                            images.append(image_tensor)
                    except Exception as e:
                        print(f"Error processing frame {frame_file}: {e}")
                        continue

                if not images:
                    raise ValueError("No frames extracted from video")

                # 合并所有帧
                if len(images) == 1:
                    result_images = images[0]
                else:
                    # 检查并调整不同尺寸的帧
                    base_image = images[0]
                    for i in range(1, len(images)):
                        if base_image.shape[1:] != images[i].shape[1:]:
                            # 调整大小以匹配
                            images[i] = comfy.utils.common_upscale(
                                images[i].movedim(-1, 1),
                                base_image.shape[3],
                                base_image.shape[2],
                                "bilinear",
                                "center"
                            ).movedim(1, -1)

                    result_images = torch.cat(images, dim=0)

                return (result_images, total_frames, fps, width, height)
        except Exception as e:
            raise ValueError(f"Failed to unpack video: {e}")
