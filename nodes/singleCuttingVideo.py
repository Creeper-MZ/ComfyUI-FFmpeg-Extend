import io
import os
import subprocess
import tempfile
from datetime import datetime
from comfy_api.input import VideoInput
from comfy_api.input_impl import VideoFromFile
from ..func import validate_time_format

class SingleCuttingVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "start_time": ("STRING", {"default":"00:00:00"}),
                "end_time": ("STRING", {"default":"00:00:10"}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "single_cutting_video"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg"

    def single_cutting_video(self, video: VideoInput, start_time, end_time):
        try:
            if not validate_time_format(start_time) or not validate_time_format(end_time):
                raise ValueError("start_time or end_time is not in time format (HH:MM:SS)")

            time_format = "%H:%M:%S"
            start_dt = datetime.strptime(start_time, time_format)
            end_dt = datetime.strptime(end_time, time_format)

            if start_dt >= end_dt:
                raise ValueError("start_time must be less than end_time")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                video.save_to(tmp_input.name)
                input_path = tmp_input.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                output_path = tmp_output.name

            try:
                command = [
                    'ffmpeg', '-i', input_path,
                    '-ss', start_time, '-to', end_time,
                    '-c', 'copy',
                    '-y', output_path
                ]

                print(f"[SingleCuttingVideo] Executing FFmpeg command: {' '.join(command)}")
                result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                if result.returncode != 0:
                    print(f"[SingleCuttingVideo] FFmpeg stderr: {result.stderr.decode('utf-8')}")
                    raise ValueError(f"FFmpeg error: {result.stderr.decode('utf-8')}")
                print(f"[SingleCuttingVideo] FFmpeg completed successfully")

                # 读取输出文件到内存后立即创建VIDEO对象
                with open(output_path, 'rb') as f:
                    output_bytes = f.read()
                output_video = VideoFromFile(io.BytesIO(output_bytes))

                return (output_video,)

            finally:
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
        except Exception as e:
            raise ValueError(f"Failed to cut video: {e}")