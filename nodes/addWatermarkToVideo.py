import io
import tempfile
import subprocess
import os
from PIL import Image
from comfy_api.input import VideoInput, ImageInput
from comfy_api.input_impl import VideoFromFile
from ..func import tensor2pil


class AddWatermarkToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "watermark_image": ("IMAGE",),
                "watermark_width": ("INT", {"default": 100, "min": 10, "max": 1920, "step": 1}),
                "position_x": ("INT", {"default": 10, "min": 0, "step": 1}),
                "position_y": ("INT", {"default": 10, "min": 0, "step": 1}),
            },
            "optional": {
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video_with_watermark",)
    FUNCTION = "add_watermark"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video"

    def add_watermark(self, video: VideoInput, watermark_image: ImageInput, watermark_width, position_x, position_y, opacity=1.0, crf=18):
        try:
            video_io = io.BytesIO()
            video.save_to(video_io)
            video_io.seek(0)
            video_bytes = video_io.read()

            watermark_pil = tensor2pil(watermark_image)
            print(f"[AddWatermarkToVideo] Watermark mode: {watermark_pil.mode}, size: {watermark_pil.size}")

            if watermark_pil.mode != 'RGBA':
                print(f"[AddWatermarkToVideo] Converting {watermark_pil.mode} to RGBA")
                watermark_pil = watermark_pil.convert('RGBA')

            if opacity < 1.0:
                print(f"[AddWatermarkToVideo] Applying opacity: {opacity}")
                alpha = watermark_pil.split()[3]
                alpha = Image.eval(alpha, lambda a: int(a * opacity))
                watermark_pil.putalpha(alpha)

            aspect_ratio = watermark_pil.height / watermark_pil.width
            watermark_height = int(watermark_width * aspect_ratio)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_bytes)
                tmp_video_path = tmp_video.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_watermark:
                watermark_pil.save(tmp_watermark.name, 'PNG')
                watermark_path = tmp_watermark.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                output_path = tmp_output.name

            try:
                print(f"[AddWatermarkToVideo] Using FFmpeg to add watermark")

                cmd = [
                    'ffmpeg',
                    '-i', tmp_video_path,
                    '-i', watermark_path,
                    '-filter_complex', f'[1:v]scale={watermark_width}:{watermark_height}[wm];[0:v][wm]overlay={position_x}:{position_y}',
                    '-c:v', 'libx264',
                    '-crf', str(crf),
                    '-preset', 'slow',
                    '-c:a', 'copy',
                    '-y',
                    output_path
                ]

                result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

                if result.returncode != 0:
                    print(f"[AddWatermarkToVideo] FFmpeg stderr: {result.stderr.decode('utf-8')}")
                    raise ValueError(f"FFmpeg error: {result.stderr.decode('utf-8')}")

                print(f"[AddWatermarkToVideo] FFmpeg completed successfully")

                with open(output_path, 'rb') as f:
                    output_bytes = f.read()

                output_video = VideoFromFile(io.BytesIO(output_bytes))

                return (output_video,)

            finally:
                for path in [tmp_video_path, watermark_path, output_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass

        except Exception as e:
            raise ValueError(f"Failed to add watermark to video: {e}")
