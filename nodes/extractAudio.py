import os
import subprocess
from ..func import video_type, get_video_bytes_from_input, extract_audio_from_bytes
from ..video_types import AudioData, video_or_string

class ExtractAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (video_or_string, {"default":"C:/Users/Desktop/video.mp4"}),
                "output_path": ("STRING", {"default":"C:/Users/Desktop/output",}),
                "audio_format": ([".m4a",".mp3",".wav",".aac",".flac",".wma",".ogg",".ac3",".amr",".aiff",".opus",".m4b",".caf",".dts"], {"default":".m4a",}),
            },
            "optional": {
                "return_audio_data": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("audio_complete_path", "audio")
    FUNCTION = "extract_audio"
    OUTPUT_NODE = True
    CATEGORY = "🔥FFmpeg"
  
    def extract_audio(self, video, output_path, audio_format, return_audio_data=False):
        try:
            output_path = os.path.abspath(output_path).strip()

            if not os.path.isdir(output_path):
                raise ValueError("output_path："+output_path+"不是目录（output_path:"+output_path+" is not a directory）")

            video_bytes = get_video_bytes_from_input(video)

            if isinstance(video, str):
                file_name = os.path.splitext(os.path.basename(video))[0]
            else:
                file_name = "audio"

            output_audio_path = os.path.join(output_path, file_name + audio_format)

            # 支持的音频格式
            supported_formats = {
                ".m4a": "m4a",
                ".mp3": "mp3",
                ".wav": "wav",
                ".aac": "aac",
                ".flac": "flac",
                ".wma": "wma",
                ".ogg": "ogg",
                ".ac3": "ac3",
                ".amr": "amr",
                ".aiff": "aiff",
                ".opus": "opus",
                ".m4b": "m4b",
                ".caf": "caf",
                ".dts": "dts"
            }

            if audio_format not in supported_formats:
                raise ValueError("不支持的音频格式："+audio_format+"(Unsupported audio formats:"+audio_format+")")

            audio_format_str = supported_formats[audio_format]

            # 提取音频
            audio_bytes = extract_audio_from_bytes(video_bytes, audio_format_str)

            # 保存音频文件
            with open(output_audio_path, 'wb') as f:
                f.write(audio_bytes)

            # 如果需要返回AudioData对象
            audio_data = None
            if return_audio_data:
                audio_data = AudioData(audio_bytes, {'format': audio_format_str})

            return (output_audio_path, audio_data)
        except Exception as e:
            raise ValueError(e)