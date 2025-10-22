"""
GetVideoBase64 节点 - 从base64字符串创建VideoData对象，并可选地提取AudioData
"""

import base64
from ..video_types import VideoData, AudioData
from ..func import get_video_info_from_bytes, extract_audio_from_bytes


class GetVideoBase64:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_base64": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "extract_audio": ("BOOLEAN", {"default": False}),
                "audio_format": ([".mp3", ".wav", ".aac", ".flac"], {"default": ".mp3"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "get_video_base64"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def get_video_base64(self, video_base64, extract_audio=False, audio_format=".mp3"):
        try:
            # 解码base64字符串为字节
            video_bytes = base64.b64decode(video_base64)

            # 获取视频信息
            video_info = get_video_info_from_bytes(video_bytes)

            # 创建VideoData对象
            video_data = VideoData(video_bytes, video_info)

            # 可选：提取音频
            audio_data = None
            if extract_audio:
                try:
                    audio_format_str = audio_format.lstrip('.')
                    audio_bytes = extract_audio_from_bytes(video_bytes, audio_format_str)
                    audio_data = AudioData(audio_bytes, {"format": audio_format_str})
                except Exception as e:
                    print(f"Warning: Failed to extract audio: {e}")

            return (video_data, audio_data)
        except Exception as e:
            raise ValueError(f"Failed to process video base64: {e}")
