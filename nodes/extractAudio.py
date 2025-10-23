import os
import av
from comfy_api.input import VideoInput


class ExtractAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "output_path": ("STRING", {"default": "C:/Users/Desktop/output"}),
                "audio_format": ([".m4a", ".mp3", ".wav", ".aac", ".flac"], {"default": ".mp3"}),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("audio_path", "audio")
    FUNCTION = "extract_audio"
    OUTPUT_NODE = True
    CATEGORY = "🔥FFmpeg"

    def extract_audio(self, video: VideoInput, output_path, audio_format):
        try:
            output_path = os.path.abspath(output_path).strip()

            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)

            # 使用官方get_components获取audio
            components = video.get_components()
            audio = components.audio

            if audio is None:
                raise ValueError("Video has no audio track")

            output_audio_path = os.path.join(output_path, f"audio{audio_format}")

            # 格式映射
            codec_map = {
                ".mp3": "libmp3lame",
                ".aac": "aac",
                ".wav": "pcm_s16le",
                ".flac": "flac",
                ".m4a": "aac"
            }

            if audio_format not in codec_map:
                raise ValueError(f"Unsupported audio format: {audio_format}")

            codec = codec_map[audio_format]
            container_format = audio_format.lstrip('.')
            if container_format == "m4a":
                container_format = "mp4"

            # 保存audio使用PyAV
            waveform = audio['waveform']  # shape: (1, channels, samples)
            sample_rate = audio['sample_rate']
            audio_data = waveform.squeeze(0).cpu().contiguous().numpy()

            print(f"[ExtractAudio] Saving audio to {output_audio_path} (format: {container_format}, codec: {codec})")
            with av.open(output_audio_path, mode='w', format=container_format) as container:
                stream = container.add_stream(codec, rate=sample_rate)
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
            print(f"[ExtractAudio] Audio saved successfully")

            return (output_audio_path, audio)
        except Exception as e:
            raise ValueError(f"Failed to extract audio: {e}")