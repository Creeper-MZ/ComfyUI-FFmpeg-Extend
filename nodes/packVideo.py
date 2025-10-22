"""
PackVideo 节点 - 将ComfyUI的IMAGE张量打包成VideoData
"""

from ..video_types import VideoData
from ..func import pack_images_to_video_bytes


class PackVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 1.0,
                    "display": "number"
                }),
            },
            "optional": {
                "output_format": (["mp4", "avi", "mov", "mkv"], {"default": "mp4"}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "pack_video"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def pack_video(self, images, fps, output_format="mp4"):
        try:
            # 验证输入
            if images is None or images.shape[0] == 0:
                raise ValueError("Input images cannot be empty")

            # 打包图像为视频字节
            video_bytes = pack_images_to_video_bytes(images, fps, output_format)

            # 创建VideoData对象
            # IMAGE张量 shape: (batch, height, width, channels)
            video_metadata = {
                'fps': fps,
                'frame_count': images.shape[0],  # batch size = frame count
                'format': output_format,
                'width': images.shape[2],  # width
                'height': images.shape[1]  # height
            }
            video_data = VideoData(video_bytes, video_metadata)

            return (video_data,)
        except Exception as e:
            raise ValueError(f"Failed to pack video: {e}")
