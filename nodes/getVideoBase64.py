import base64
import io
from comfy_api.input_impl import VideoFromFile


class GetVideoBase64:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_base64": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "get_video_base64"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def get_video_base64(self, video_base64):
        try:
            video_bytes = base64.b64decode(video_base64)
            video_io = io.BytesIO(video_bytes)
            video = VideoFromFile(video_io)
            return (video,)
        except Exception as e:
            raise ValueError(f"Failed to process video base64: {e}")
