import base64
import io
from comfy_api.input import VideoInput
from comfy_api.util import VideoContainer, VideoCodec


class VideoToBase64:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "format": (VideoContainer.as_input(), {"default": "mp4"}),
                "codec": (VideoCodec.as_input(), {"default": "h264"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_base64",)
    FUNCTION = "video_to_base64"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def video_to_base64(self, video: VideoInput, format="mp4", codec="h264"):
        try:
            video_io = io.BytesIO()
            video.save_to(
                video_io,
                format=VideoContainer(format),
                codec=VideoCodec(codec)
            )
            video_io.seek(0)
            video_bytes = video_io.read()
            base64_string = base64.b64encode(video_bytes).decode('utf-8')
            return (base64_string,)
        except Exception as e:
            raise ValueError(f"Failed to convert video to base64: {e}")
