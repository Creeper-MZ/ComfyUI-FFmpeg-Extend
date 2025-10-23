import os
import torch
import numpy as np
from PIL import Image
from comfy_api.input import VideoInput
import av


class Video2Frames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "output_path": ("STRING", {"default": "C:/Users/Desktop/output"}),
                "frames_max_width": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("frame_path", "fps", "audio_path", "total_frames", "output_path")
    FUNCTION = "video2frames"
    OUTPUT_NODE = True
    CATEGORY = "🔥FFmpeg"

    def video2frames(self, video: VideoInput, output_path, frames_max_width):
        try:
            output_path = os.path.abspath(output_path).strip()

            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)

            # 使用官方get_components获取视频组件
            components = video.get_components()
            images = components.images  # shape: (frames, height, width, channels)
            audio = components.audio
            fps = float(components.frame_rate)

            total_frames = images.shape[0]
            orig_height = images.shape[1]
            orig_width = images.shape[2]

            # 保存帧
            frame_path = os.path.join(output_path, 'frames')
            os.makedirs(frame_path, exist_ok=True)

            for i in range(total_frames):
                frame_tensor = images[i]  # shape: (height, width, channels)

                # 缩放处理
                if frames_max_width > 0 and orig_width > frames_max_width:
                    scale_factor = frames_max_width / orig_width
                    new_height = int(orig_height * scale_factor)
                    frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(frame_np)
                    pil_img = pil_img.resize((frames_max_width, new_height), Image.LANCZOS)
                else:
                    frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(frame_np)

                pil_img.save(os.path.join(frame_path, f'frame_{i:08d}.png'))

            # 保存音频
            audio_path = ""
            if audio is not None:
                audio_path = os.path.join(output_path, 'audio.mp3')
                try:
                    waveform = audio['waveform']  # shape: (1, channels, samples)
                    sample_rate = audio['sample_rate']

                    audio_data = waveform.squeeze(0).cpu().contiguous().numpy()

                    print(f"[Video2Frames] Saving audio to {audio_path}")
                    with av.open(audio_path, mode='w', format='mp3') as container:
                        stream = container.add_stream('libmp3lame', rate=sample_rate)
                        frame = av.AudioFrame.from_ndarray(
                            audio_data,
                            format='fltp',
                            layout='stereo' if audio_data.shape[0] > 1 else 'mono'
                        )
                        frame.sample_rate = sample_rate
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        for packet in stream.encode(None):
                            container.mux(packet)
                    print(f"[Video2Frames] Audio saved successfully")
                except Exception as e:
                    print(f"[Video2Frames] Warning: Failed to save audio: {e}")
                    audio_path = ""

            print(f"提取完成: fps={fps}, frames={total_frames}, size={orig_width}x{orig_height}")

            return (frame_path, fps, audio_path, total_frames, output_path)
        except Exception as e:
            raise ValueError(f"Failed to extract frames: {e}")