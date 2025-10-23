import numpy as np
from PIL import Image
import torch
import subprocess
import json
import re
import os
import gc
import shutil
import time
import glob
import base64
import io
import tempfile
import cv2
import av
from itertools import islice
from concurrent.futures import ThreadPoolExecutor,as_completed
from comfy.model_management import unload_all_models, soft_empty_cache

def get_xfade_transitions():
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-h', 'filter=xfade'],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout if result.stdout else result.stderr
        print(output)
        pattern = r'^\s*(\w+)\s+-?\d+\b'
        data = output.split('\n')
        if len(data) == 0:
            transitions = [
                'fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown',
                'slideleft', 'slideright', 'slideup', 'slidedown',
                'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite',
                'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown',
                'circleopen', 'circleclose', 'vertopen', 'vertclose',
                'horzopen', 'horzclose', 'dissolve', 'pixelize',
                'diagtl', 'diagtr', 'diagbl', 'diagbr',
                'hlslice', 'hrslice', 'vuslice', 'vdslice',
                'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr',
                'squeezeh', 'squeezev', 'zoomin', 'fadefast', 'fadeslow',
                'hlwind', 'hrwind', 'vuwind', 'vdwind',
                'coverleft', 'coverright', 'coverup', 'coverdown',
                'revealleft', 'revealright', 'revealup', 'revealdown'
            ]
        else:
            transitions = []
            for line in data:
                match = re.search(pattern, line)
                if match and match.group(1) != 'none' and match.group(1) != 'custom':
                    transitions.append(match.group(1))
                
        return sorted(transitions)
    
    except subprocess.CalledProcessError as e:
        print(f"执行ffmpeg命令出错: {e}")
        print(f"错误输出: {e.stderr}")
        return []
    except FileNotFoundError:
        print("错误: 找不到ffmpeg程序，请确保ffmpeg已安装并添加到系统PATH")
        return []

def copy_image(image_path, destination_directory):
    try:
        image_name = os.path.basename(image_path)
        destination_path = os.path.join(destination_directory, image_name)
        if not os.path.exists(destination_path):
            shutil.copy(image_path, destination_path)
        return destination_path
    except Exception as e:
        print(f"Error copying image {image_path}: {e}")
        return None

def copy_images_to_directory(image_paths, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    index_to_path = {i: image_path for i, image_path in enumerate(image_paths)}
    copied_paths = [None] * len(image_paths)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(copy_image, image_path, destination_directory): i for i, image_path in index_to_path.items()}

        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            if result is not None:
                copied_paths[index] = result

    return [path for path in copied_paths if path is not None]

def get_image_paths_from_directory(directory, start_index, length):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    def image_generator():
        for filename in sorted(os.listdir(directory)):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                yield os.path.join(directory, filename)

    selected_images = islice(image_generator(), start_index, start_index + length)

    return list(selected_images)

def generate_template_string(filename):
    match = re.search(r'\d+', filename)
    return re.sub(r'\d+', lambda x: f'%0{len(x.group())}d', filename) if match else filename

def tensor2pil(image):
    img_np = image.cpu().numpy().squeeze()

    if len(img_np.shape) == 2:
        img_np = np.clip(255. * img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='L')
    elif img_np.shape[2] == 3:
        img_np = np.clip(255. * img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='RGB')
    elif img_np.shape[2] == 4:
        img_np = np.clip(255. * img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='RGBA')
    else:
        img_np = np.clip(255. * img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def getVideoInfo(video_path):
    command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=avg_frame_rate,duration,width,height', '-of', 'json', video_path
        ]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    print(output)
    data = json.loads(output)
    if 'streams' in data and len(data['streams']) > 0:
        stream = data['streams'][0]
        fps = stream.get('avg_frame_rate')
        if fps is not None:
            if '/' in fps:
                num, denom = map(int, fps.split('/'))
                fps = num / denom
            else:
                fps = float(fps)
            width = int(stream.get('width'))
            height = int(stream.get('height'))
            duration = float(stream.get('duration'))
            return_data = {'fps': fps, 'width': width, 'height': height, 'duration': duration}
    else:
        return_data = {}
    return return_data

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height
    
def has_audio(video_path):
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'a:0', 
        '-show_entries', 'stream=codec_type', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode().strip() == 'audio'

def set_file_name(video_path):
    file_name = os.path.basename(video_path)
    file_extension = os.path.splitext(file_name)[1]
    file_name = time.strftime("%Y%m%d%H%M%S", time.localtime()) + file_extension
    return file_name

def video_type():
    return ('.mp4', '.avi', '.mov', '.mkv','.rmvb','.wmv','.flv')
def audio_type():
    return ('.mp3', '.wav', '.aac', '.flac','.m4a','.wma','.ogg','.amr','.ape','.ac3','.aiff','.opus','.m4b','.caf','.dts')

def validate_time_format(time_str):
    pattern = r'^([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]|\d{1,2})$'
    return bool(re.match(pattern, time_str))

def get_video_files(directory):
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv','*.rmvb', '*.wmv', '*.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, ext)))
    video_files.sort()
    return video_files

def save_image(image, path):
    tensor2pil(image).save(path)
    
def clear_memory():
    gc.collect()
    unload_all_models()
    soft_empty_cache()

def get_video_info_from_bytes(video_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)

            if not cap.isOpened():
                raise ValueError("Unable to open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'frame_count': frame_count
            }
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to get video info from bytes: {e}")

def extract_frames_from_bytes(video_bytes, output_dir, fps_scale=None, use_ffmpeg_pipe=True):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            os.makedirs(output_dir, exist_ok=True)

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Unable to open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if fps_scale and fps_scale > 0 and fps_scale != 1.0:
                scaled_width = int(width * fps_scale)
                scaled_height = int(height * fps_scale)
            else:
                scaled_width = width
                scaled_height = height

            if use_ffmpeg_pipe:
                frame_count = _extract_frames_ffmpeg_pipe(
                    tmp_path, output_dir, scaled_width, scaled_height, fps_scale
                )
            else:
                frame_count = _extract_frames_opencv(
                    tmp_path, output_dir, scaled_width, scaled_height, fps_scale
                )

            return frame_count, fps, scaled_width, scaled_height
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to extract frames from bytes: {e}")


def _extract_frames_ffmpeg_pipe(video_path, output_dir, scaled_width, scaled_height, fps_scale=None):
    scale_filter = f"scale={scaled_width}:{scaled_height}" if fps_scale and fps_scale != 1.0 else ""

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
    ]

    if scale_filter:
        cmd.extend(['-vf', scale_filter])

    cmd.append('-')

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        frame_count = 0
        frame_bytes = scaled_width * scaled_height * 3

        while True:
            data = process.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                break

            frame_array = np.frombuffer(data, dtype=np.uint8).reshape((scaled_height, scaled_width, 3))

            frame_path = os.path.join(output_dir, f'frame_{frame_count:08d}.png')
            pil_image = Image.fromarray(frame_array, 'RGB')
            pil_image.save(frame_path)

            frame_count += 1

        process.wait()
        return frame_count
    except Exception as e:
        raise ValueError(f"Failed to extract frames using FFmpeg pipe: {e}")


def _extract_frames_opencv(video_path, output_dir, scaled_width, scaled_height, fps_scale=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fps_scale and fps_scale != 1.0:
            frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_path = os.path.join(output_dir, f'frame_{frame_count:08d}.png')
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(frame_path)

        frame_count += 1

    cap.release()
    return frame_count

def pack_images_to_video_bytes(images, fps, output_format='mp4'):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}') as tmp:
            output_file = tmp.name

        try:
            height = images.shape[1]
            width = images.shape[2]

            if output_format == 'mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif output_format == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            elif output_format == 'mov':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            if not out.isOpened():
                raise ValueError(f"Failed to create video writer for format: {output_format}")

            for i in range(images.shape[0]):
                frame_tensor = images[i:i+1]

                frame_pil = tensor2pil(frame_tensor)
                frame_np = np.array(frame_pil)

                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                out.write(frame_bgr)

            out.release()

            with open(output_file, 'rb') as f:
                video_bytes = f.read()

            return video_bytes
        finally:
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to pack images to video bytes: {e}")

def extract_audio_from_bytes(video_bytes, audio_format='mp3'):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}') as tmp_audio:
            tmp_audio_path = tmp_audio.name

        try:
            input_container = av.open(tmp_video_path)

            if not input_container.audio:
                raise ValueError("Video has no audio stream")

            if audio_format == 'mp3':
                codec_name = 'libmp3lame'
            elif audio_format == 'aac':
                codec_name = 'aac'
            elif audio_format == 'wav':
                codec_name = 'pcm_s16le'
            else:
                codec_name = 'libmp3lame'

            output_container = av.open(tmp_audio_path, 'w')

            audio_stream = input_container.audio[0]
            out_stream = output_container.add_stream(codec_name, rate=audio_stream.sample_rate)

            for frame in input_container.decode(audio=0):
                for packet in out_stream.encode(frame):
                    output_container.mux(packet)

            for packet in out_stream.encode():
                output_container.mux(packet)

            input_container.close()
            output_container.close()

            with open(tmp_audio_path, 'rb') as f:
                audio_bytes = f.read()

            return audio_bytes
        finally:
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except:
                    pass
            if os.path.exists(tmp_audio_path):
                try:
                    os.remove(tmp_audio_path)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to extract audio from bytes: {e}")

def add_watermark_to_video_bytes(video_bytes, watermark_image, watermark_width, position_x, position_y):
    try:
        if isinstance(watermark_image, str):
            if not os.path.exists(watermark_image):
                raise ValueError(f"Watermark image not found: {watermark_image}")
            watermark_pil = Image.open(watermark_image).convert('RGBA')
        elif isinstance(watermark_image, Image.Image):
            watermark_pil = watermark_image.convert('RGBA')
        else:
            raise ValueError("Watermark image must be a file path or PIL Image")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_watermark:
            watermark_pil.save(tmp_watermark.name)
            watermark_path = tmp_watermark.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
            output_path = tmp_output.name

        try:
            command = [
                'ffmpeg',
                '-i', tmp_video_path,
                '-i', watermark_path,
                '-filter_complex',
                f'[1:v]scale={watermark_width}:-1[wm];[0:v][wm]overlay={position_x}:{position_y}',
                '-c:v', 'libx264',
                '-crf', '0',
                '-preset', 'slow',
                '-c:a', 'copy',
                '-y',
                output_path
            ]

            print(f"[AddWatermark] Executing FFmpeg command: {' '.join(command)}")
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if result.returncode != 0:
                print(f"[AddWatermark] FFmpeg stderr: {result.stderr.decode('utf-8')}")
                raise ValueError(f"FFmpeg error: {result.stderr.decode('utf-8')}")
            print(f"[AddWatermark] FFmpeg completed successfully")

            with open(output_path, 'rb') as f:
                output_bytes = f.read()

            return output_bytes
        finally:
            for path in [tmp_video_path, watermark_path, output_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
    except Exception as e:
        raise ValueError(f"Failed to add watermark to video: {e}")

def get_video_bytes_from_input(video_input):
    from .video_types import VideoData

    if isinstance(video_input, VideoData):
        return video_input.to_bytes()
    elif isinstance(video_input, str):
        if not os.path.exists(video_input):
            raise ValueError(f"Video file not found: {video_input}")
        with open(video_input, 'rb') as f:
            return f.read()
    else:
        raise ValueError("Video input must be a file path (STRING) or VideoData object")

def get_audio_bytes_from_input(audio_input):
    from .video_types import AudioData

    if isinstance(audio_input, AudioData):
        return audio_input.to_bytes()
    elif isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        with open(audio_input, 'rb') as f:
            return f.read()
    else:
        raise ValueError("Audio input must be a file path (STRING) or AudioData object")

def add_watermark_to_frame(frame, watermark, x, y):
    frame_height, frame_width = frame.shape[:2]
    wm_height, wm_width = watermark.shape[:2]

    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    x_end = min(x + wm_width, frame_width)
    y_end = min(y + wm_height, frame_height)

    wm_w_adjusted = x_end - x
    wm_h_adjusted = y_end - y

    if wm_w_adjusted <= 0 or wm_h_adjusted <= 0:
        return frame

    watermark_region = watermark[:wm_h_adjusted, :wm_w_adjusted]

    if watermark_region.shape[2] == 4:
        watermark_rgb = watermark_region[:, :, :3]
        watermark_alpha = watermark_region[:, :, 3] / 255.0
    else:
        watermark_rgb = watermark_region
        watermark_alpha = np.ones((wm_h_adjusted, wm_w_adjusted))

    watermark_bgr = cv2.cvtColor(watermark_rgb, cv2.COLOR_RGB2BGR)

    frame_roi = frame[y:y_end, x:x_end]

    for c in range(3):
        frame_roi[:, :, c] = frame_roi[:, :, c] * (1 - watermark_alpha) + watermark_bgr[:, :, c] * watermark_alpha

    frame[y:y_end, x:x_end] = frame_roi

    return frame

def add_watermark_to_video_bytes_pyav(video_bytes, watermark_image, watermark_width, position_x, position_y):
    try:
        if isinstance(watermark_image, str):
            if not os.path.exists(watermark_image):
                raise ValueError(f"Watermark image not found: {watermark_image}")
            watermark_pil = Image.open(watermark_image).convert('RGBA')
        elif isinstance(watermark_image, Image.Image):
            watermark_pil = watermark_image.convert('RGBA')
        else:
            raise ValueError("Watermark image must be a file path or PIL Image")

        aspect_ratio = watermark_pil.height / watermark_pil.width
        watermark_height = int(watermark_width * aspect_ratio)
        watermark_pil = watermark_pil.resize((watermark_width, watermark_height), Image.LANCZOS)

        watermark_np = np.array(watermark_pil)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
            tmp_input.write(video_bytes)
            tmp_input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            input_container = av.open(tmp_input_path)
            input_stream = input_container.streams.video[0]

            output_container = av.open(tmp_output_path, 'w')

            output_stream = output_container.add_stream('libx264', rate=input_stream.average_rate)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = 'yuv420p'
            output_stream.options = {'crf': '0', 'preset': 'slow'}

            audio_stream = None
            if len(input_container.streams.audio) > 0:
                input_audio = input_container.streams.audio[0]
                audio_stream = output_container.add_stream(input_audio.codec_context.name, rate=input_audio.sample_rate)

            print(f"[AddWatermark PyAV] Processing video: {input_stream.width}x{input_stream.height}, {input_stream.frames} frames")

            frame_count = 0
            for packet in input_container.demux():
                if packet.stream.type == 'video':
                    for frame in packet.decode():
                        img = frame.to_ndarray(format='bgr24')

                        img = add_watermark_to_frame(img, watermark_np, position_x, position_y)

                        new_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                        new_frame.pts = frame.pts
                        new_frame.time_base = frame.time_base

                        for out_packet in output_stream.encode(new_frame):
                            output_container.mux(out_packet)

                        frame_count += 1
                        if frame_count % 30 == 0:
                            print(f"[AddWatermark PyAV] Processed {frame_count} frames...")

                elif packet.stream.type == 'audio' and audio_stream:
                    packet.stream = audio_stream
                    output_container.mux(packet)

            for packet in output_stream.encode():
                output_container.mux(packet)

            print(f"[AddWatermark PyAV] Completed processing {frame_count} frames")

            input_container.close()
            output_container.close()

            with open(tmp_output_path, 'rb') as f:
                output_bytes = f.read()

            return output_bytes

        finally:
            for path in [tmp_input_path, tmp_output_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

    except Exception as e:
        raise ValueError(f"Failed to add watermark using PyAV: {e}")