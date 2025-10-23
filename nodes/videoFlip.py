import io
import os
import subprocess
import tempfile
from comfy_api.input import VideoInput
from comfy_api.input_impl import VideoFromFile

class VideoFlip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "flip_type": (["horizontal","vertical","both"], {"default":"horizontal"}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "video_flip"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg"

    def video_flip(self, video: VideoInput, flip_type):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                video.save_to(tmp_input.name)
                input_path = tmp_input.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                output_path = tmp_output.name

            try:
                flip = {
                    'horizontal': 'hflip',
                    'vertical': 'vflip',
                    'both': 'hflip,vflip',
                }.get(flip_type, 'hflip')

                command = [
                    'ffmpeg', '-i', input_path,
                    '-vf', flip,
                    '-c:a', 'copy',
                    '-y', output_path,
                ]

                print(f"[VideoFlip] Executing FFmpeg command: {' '.join(command)}")
                result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                if result.returncode != 0:
                    print(f"[VideoFlip] FFmpeg stderr: {result.stderr.decode('utf-8')}")
                    raise ValueError(f"FFmpeg error: {result.stderr.decode('utf-8')}")
                print(f"[VideoFlip] FFmpeg completed successfully")

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
            raise ValueError(f"Failed to flip video: {e}")