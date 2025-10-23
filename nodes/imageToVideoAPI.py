import io
import time
import base64
import json
import requests
from PIL import Image
import numpy as np
from comfy_api.input_impl import VideoFromFile

class ImageToVideoAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "The character is smiling and waving gently",
                    "multiline": True
                }),
                "api_key": ("STRING", {
                    "default": "",
                }),
                "model": (["Pro", "Fast"], {
                    "default": "Pro"
                }),
                "resolution": (["480p", "720p", "1080p"], {
                    "default": "720p"
                }),
                "ratio": (["from_input", "16:9", "9:16", "1:1", "3:4", "4:3", "21:9"], {
                    "default": "from_input"
                }),
                "duration": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 10,
                    "step": 1
                }),
                "framespersecond": ("INT", {
                    "default": 24,
                    "min": 12,
                    "max": 60,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
                "camera_fixed": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "end_frame_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "raw_response")
    FUNCTION = "generate_video"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Video/API"

    MODEL_MAPPING = {
        'Pro': 'ep-20251019161741-f85ph',
        'Fast': 'ep-20251019201232-pxvvd'
    }

    def _get_ratio_from_image(self, image_tensor):
        height = image_tensor.shape[1]
        width = image_tensor.shape[2]
        ratio = f"{width}:{height}"
        print(f"[ImageToVideoAPI] Input image size: {width}x{height}, using exact ratio: {ratio}")
        return ratio

    def _image_to_base64(self, image_tensor):
        img_array = image_tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        return f"data:image/jpeg;base64,{base64.b64encode(img_buffer.getvalue()).decode('utf-8')}"

    def generate_video(self, image, prompt, api_key, model, resolution, ratio, duration, framespersecond, seed, camera_fixed, end_frame_image=None):
        try:
            if not api_key:
                raise ValueError("API key is required. Please provide your Bytedance API token.")

            source_image_base64 = self._image_to_base64(image)

            if ratio == "from_input":
                ratio = self._get_ratio_from_image(image)

            endpoint_id = self.MODEL_MAPPING[model]
            api_url = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"

            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": source_image_base64},
                    "role": "first_frame"
                }
            ]

            if end_frame_image is not None:
                end_frame_base64 = self._image_to_base64(end_frame_image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": end_frame_base64},
                    "role": "last_frame"
                })

                if model == "Fast":
                    text_params = f"--workflow image2video-i2v --camera_model none --resolution {resolution} --ratio {ratio} --duration {duration} --fps {framespersecond} --seed {seed} --camerafixed {str(camera_fixed).lower()}"
                    if prompt:
                        text_params = f"{prompt} {text_params}"
                else:
                    print("[ImageToVideoAPI] Warning: Pro model does not support end frame, ignoring end_frame_image")
                    text_params = f"{prompt} --resolution {resolution} --ratio {ratio} --duration {duration} --fps {framespersecond} --seed {seed} --camerafixed {str(camera_fixed).lower()}"
            else:
                text_params = f"{prompt} --resolution {resolution} --ratio {ratio} --duration {duration} --fps {framespersecond} --seed {seed} --camerafixed {str(camera_fixed).lower()}"

            content.append({
                "type": "text",
                "text": text_params
            })

            request_body = {
                "model": endpoint_id,
                "content": content
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            mode_str = "with end frame" if end_frame_image is not None and model == "Fast" else "single image"
            print(f"[ImageToVideoAPI] Sending request to Bytedance API with model: {model} ({mode_str})")
            print(f"[ImageToVideoAPI] Prompt: {prompt}")

            response = requests.post(api_url, json=request_body, headers=headers)

            if response.status_code != 200:
                raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

            result = response.json()
            task_id = result.get('id')

            if not task_id:
                raise ValueError(f"Failed to get task ID from response: {result}")

            print(f"[ImageToVideoAPI] Task created with ID: {task_id}")

            estimated_time = "3-10 minutes" if model == "Pro" else "1-3 minutes"
            print(f"[ImageToVideoAPI] Polling for task completion (this may take {estimated_time})...")

            max_attempts = 120
            base_interval = 10

            for attempt in range(max_attempts):
                time.sleep(base_interval)

                poll_url = f"{api_url}/{task_id}"
                poll_response = requests.get(poll_url, headers=headers)

                if poll_response.status_code != 200:
                    print(f"[ImageToVideoAPI] Poll attempt {attempt + 1}/{max_attempts}: Status check failed")
                    continue

                task_result = poll_response.json()
                status = task_result.get('status')

                print(f"[ImageToVideoAPI] Poll attempt {attempt + 1}/{max_attempts}: Status = {status}")

                if status in ['completed', 'succeeded']:
                    video_url = task_result.get('content', {}).get('video_url')

                    if not video_url:
                        raise ValueError(f"Task completed but no video URL found: {task_result}")

                    print(f"[ImageToVideoAPI] Video generation completed! Downloading from: {video_url}")

                    video_response = requests.get(video_url)

                    if video_response.status_code != 200:
                        raise ValueError(f"Failed to download video: {video_response.status_code}")

                    video_bytes = video_response.content
                    video = VideoFromFile(io.BytesIO(video_bytes))

                    print(f"[ImageToVideoAPI] Video downloaded successfully, size: {len(video_bytes) / 1024 / 1024:.2f}MB")

                    raw_response = json.dumps(task_result, indent=2, ensure_ascii=False)

                    return (video, raw_response)

                elif status == 'failed':
                    raise ValueError(f"Task failed: {task_result}")

            raise ValueError(f"Task timeout after {max_attempts} attempts")

        except Exception as e:
            raise ValueError(f"Failed to generate video from image: {e}")
