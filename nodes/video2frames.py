import json
import math
import os
import subprocess
from ..func import video_type, get_video_bytes_from_input, extract_frames_from_bytes, extract_audio_from_bytes, get_video_info_from_bytes
from ..video_types import VideoData, video_or_string

class Video2Frames:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (video_or_string, {"default":"C:/Users/Desktop/video.mp4"}),
                "output_path": ("STRING", {"default":"C:/Users/Desktop/output",}),
                "frames_max_width":("INT", {"default": 0, "min": 0, "max": 1920}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "INT","STRING")
    RETURN_NAMES = ("frame_path", "fps", "audio_path", "total_frames","output_path")
    FUNCTION = "video2frames"
    OUTPUT_NODE = True
    CATEGORY = "🔥FFmpeg"

    def video2frames(self, video, output_path, frames_max_width):
        try:
            output_path = os.path.abspath(output_path).strip()

            # 判断output_path是否是一个目录
            if not os.path.isdir(output_path):
                raise ValueError("output_path："+output_path+"不是目录（output_path:"+output_path+" is not a directory）")

            # 判断frames_max_width是否是一个整数
            if not isinstance(frames_max_width, int):
                raise ValueError("frames_max_width不是整数（frames_max_width is not an integer）")

            # 获取视频字节数据（支持VideoData或文件路径）
            video_bytes = get_video_bytes_from_input(video)

            # 获取视频信息
            video_info = get_video_info_from_bytes(video_bytes)
            fps = video_info['fps']
            width = video_info['width']
            height = video_info['height']

            # 计算输出宽度和高度以保持比例
            if frames_max_width > 0:
                if width > frames_max_width:
                    out_width = frames_max_width
                    out_height = int(height * frames_max_width / width)
                else:
                    out_width = width
                    out_height = height
            else:
                out_width = width
                out_height = height

            # 计算缩放因子
            scale_factor = out_width / width if width > 0 else 1.0

            # 提取帧
            frame_path = os.path.join(output_path, 'frames')
            os.makedirs(frame_path, exist_ok=True)

            total_frames, fps_out, width_out, height_out = extract_frames_from_bytes(
                video_bytes,
                frame_path,
                fps_scale=scale_factor if scale_factor != 1.0 else None
            )

            # 提取音频
            audio_path = os.path.join(output_path, 'audio.mp3')
            try:
                audio_bytes = extract_audio_from_bytes(video_bytes, 'mp3')
                with open(audio_path, 'wb') as f:
                    f.write(audio_bytes)
            except Exception as e:
                print(f"Warning: Failed to extract audio: {e}")
                audio_path = ""

            print(f"视频的帧率是: {fps_out}, 宽度是: {width_out}, 高度是: {height_out}, 总帧数是: {total_frames}")

            return (frame_path, fps_out, audio_path, total_frames, output_path)
        except Exception as e:
            raise ValueError(e)