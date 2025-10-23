import json
from comfy_api.input import VideoInput


class GetVideoInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("fps", "width", "height", "duration", "info_json")
    FUNCTION = "get_video_info"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def get_video_info(self, video: VideoInput):
        try:
            width, height = video.get_dimensions()
            duration = video.get_duration()
            components = video.get_components()
            fps = float(components.frame_rate)
            frame_count = components.images.shape[0]

            info_json = json.dumps({
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'frame_count': frame_count,
            }, ensure_ascii=False, indent=2)

            return (fps, width, height, duration, info_json)
        except Exception as e:
            raise ValueError(f"Failed to get video info: {e}")
