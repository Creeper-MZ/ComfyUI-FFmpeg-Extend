import io
import os
import subprocess
import tempfile
from comfy_api.input import VideoInput
from comfy_api.input_impl import VideoFromFile

class VideoPlayback:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "reverse_audio": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "video_playback"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg"

    def video_playback(self, video: VideoInput, reverse_audio):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                video.save_to(tmp_input.name)
                input_path = tmp_input.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                output_path = tmp_output.name

            try:
                command = ['ffmpeg', '-i', input_path, '-vf', 'reverse']

                if reverse_audio:
                    command.extend(['-af', 'areverse'])

                command.extend(['-y', output_path])

                print(f"[VideoPlayback] Executing FFmpeg command: {' '.join(command)}")
                result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                if result.returncode != 0:
                    print(f"[VideoPlayback] FFmpeg stderr: {result.stderr.decode('utf-8')}")
                    raise ValueError(f"FFmpeg error: {result.stderr.decode('utf-8')}")
                print(f"[VideoPlayback] FFmpeg completed successfully")

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
            raise ValueError(f"Failed to reverse video: {e}")