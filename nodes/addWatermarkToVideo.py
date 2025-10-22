"""
AddWatermarkToVideo 节点 - 给VideoData添加图片水印
"""

from PIL import Image
import numpy as np
from ..video_types import VideoData
from ..func import add_watermark_to_video_bytes, tensor2pil


class AddWatermarkToVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "watermark_image": ("IMAGE",),
                "watermark_width": ("INT", {"default": 100, "min": 10, "max": 1920, "step": 1}),
                "position_x": ("INT", {"default": 10, "min": 0, "step": 1}),
                "position_y": ("INT", {"default": 10, "min": 0, "step": 1}),
            },
            "optional": {
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video_with_watermark",)
    FUNCTION = "add_watermark"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def add_watermark(self, video, watermark_image, watermark_width, position_x, position_y, opacity=1.0):
        try:
            # 确保输入是VideoData对象
            if not isinstance(video, VideoData):
                raise ValueError("Input video must be a VideoData object")

            # 将IMAGE张量转换为PIL Image
            watermark_pil = tensor2pil(watermark_image)

            # 应用透明度
            if opacity < 1.0:
                # 转换为RGBA以支持透明度
                watermark_pil = watermark_pil.convert('RGBA')
                # 获取或创建alpha通道
                if len(watermark_pil.split()) == 4:
                    alpha = watermark_pil.split()[3]
                else:
                    alpha = Image.new('L', watermark_pil.size, 255)
                # 应用透明度
                alpha = Image.new('L', alpha.size, int(255 * opacity))
                watermark_pil.putalpha(alpha)
                watermark_pil = watermark_pil.convert('RGBA')

            # 添加水印
            output_bytes = add_watermark_to_video_bytes(
                video.to_bytes(),
                watermark_pil,
                watermark_width,
                position_x,
                position_y
            )

            # 创建新的VideoData对象
            video_metadata = video.get_metadata().copy()
            output_video = VideoData(output_bytes, video_metadata)

            return (output_video,)
        except Exception as e:
            raise ValueError(f"Failed to add watermark to video: {e}")
