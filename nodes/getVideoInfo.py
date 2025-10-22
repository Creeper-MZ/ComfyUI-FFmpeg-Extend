"""
GetVideoInfo 节点 - 获取VideoData的视频信息
"""

import json
from ..video_types import VideoData
from ..func import get_video_info_from_bytes


class GetVideoInfo:
    def __init__(self):
        pass

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

    def get_video_info(self, video):
        try:
            # 确保输入是VideoData对象
            if not isinstance(video, VideoData):
                raise ValueError("Input must be a VideoData object")

            # 从VideoData元信息或直接获取视频信息
            video_info = video.get_metadata()

            # 如果元信息不完整，则从字节重新计算
            if not all(k in video_info for k in ['fps', 'width', 'height', 'duration']):
                video_info = get_video_info_from_bytes(video.to_bytes())
                video.set_metadata(video_info)

            fps = float(video_info.get('fps', 0))
            width = int(video_info.get('width', 0))
            height = int(video_info.get('height', 0))
            duration = float(video_info.get('duration', 0))

            # 生成JSON字符串
            info_json = json.dumps({
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'frame_count': int(fps * duration) if fps > 0 else 0,
                'video_size_mb': video.get_size() / (1024 * 1024)
            }, ensure_ascii=False, indent=2)

            return (fps, width, height, duration, info_json)
        except Exception as e:
            raise ValueError(f"Failed to get video info: {e}")
