from fractions import Fraction
from comfy_api.input import ImageInput, AudioInput, VideoInput
from comfy_api.util import VideoComponents
from comfy_api.input_impl import VideoFromComponents
from typing import Optional


class PackVideo:
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
                "audio": ("AUDIO", {"default": None}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "pack_video"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def pack_video(self, images: ImageInput, fps: float, audio: Optional[AudioInput] = None) -> tuple[VideoInput]:
        try:
            if images is None or images.shape[0] == 0:
                raise ValueError("Input images cannot be empty")

            video = VideoFromComponents(
                VideoComponents(
                    images=images,
                    audio=audio,
                    frame_rate=Fraction(fps),
                )
            )
            return (video,)
        except Exception as e:
            raise ValueError(f"Failed to pack video: {e}")
