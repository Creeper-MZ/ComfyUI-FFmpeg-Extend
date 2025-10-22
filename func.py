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
        # 获取FFmpeg xfade过滤器支持的转场列表
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-h', 'filter=xfade'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 命令输出在stderr中
        output = result.stdout if result.stdout else result.stderr
        print(output)
        # 使用正则表达式匹配所有transition行
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
            ]  # 如果没有找到任何transition，使用默认的
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
        # 获取图片文件名
        image_name = os.path.basename(image_path)
        # 构建目标路径
        destination_path = os.path.join(destination_directory, image_name)
        # 检查目标路径是否已有相同文件，避免重复复制
        if not os.path.exists(destination_path):
            shutil.copy(image_path, destination_path)
        return destination_path
    except Exception as e:
        print(f"Error copying image {image_path}: {e}")
        return None

def copy_images_to_directory(image_paths, destination_directory):
    # 如果目标目录不存在，创建它
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # 使用字典来保持原始索引与路径的对应关系
    index_to_path = {i: image_path for i, image_path in enumerate(image_paths)}
    copied_paths = [None] * len(image_paths)

    # 使用多线程并行复制图片
    with ThreadPoolExecutor() as executor:
        # 提交所有任务
        futures = {executor.submit(copy_image, image_path, destination_directory): i for i, image_path in index_to_path.items()}
        
        # 等待所有任务完成并按顺序存储结果
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            if result is not None:
                copied_paths[index] = result

    # 返回按原始顺序排列的新路径
    return [path for path in copied_paths if path is not None]

def get_image_paths_from_directory(directory, start_index, length):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    # 创建排序后的文件生成器，直接在生成器中过滤
    def image_generator():
        for filename in sorted(os.listdir(directory)):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                yield os.path.join(directory, filename)

    # 使用islice获取所需的图像路径
    selected_images = islice(image_generator(), start_index, start_index + length)
    
    return list(selected_images)


# def get_image_paths_from_directory(directory, start_index, length):
#     # 获取目录下所有文件，并按照文件名排序
#     files = sorted(os.listdir(directory))
    
#     # 过滤掉非图片文件（这里只检查常见图片格式）
#     image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
#     image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
#     # 获取从start_index开始的length个图片路径
#     selected_images = image_files[start_index:start_index + length]
    
#     # 返回完整路径列表
#     image_paths = [os.path.join(directory, image_file) for image_file in selected_images]
    
#     return image_paths

def generate_template_string(filename):
    match = re.search(r'\d+', filename)
    return re.sub(r'\d+', lambda x: f'%0{len(x.group())}d', filename) if match else filename

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def getVideoInfo(video_path):
    command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 
            'stream=avg_frame_rate,duration,width,height', '-of', 'json', video_path
        ]
    # 运行ffprobe命令
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    # 将输出转化为字符串
    output = result.stdout.decode('utf-8').strip()
    print(output)
    data = json.loads(output)
    # 查找视频流信息
    if 'streams' in data and len(data['streams']) > 0:
        stream = data['streams'][0]  # 获取第一个视频流
        fps = stream.get('avg_frame_rate')
        if fps is not None:
            # 帧率可能是一个分数形式的字符串，例如 "30/1" 或 "20.233000"
            if '/' in fps:
                num, denom = map(int, fps.split('/'))
                fps = num / denom
            else:
                fps = float(fps)  # 直接转换为浮点数
            width = int(stream.get('width'))
            height = int(stream.get('height'))
            duration = float(stream.get('duration'))
            return_data = {'fps': fps, 'width': width, 'height': height, 'duration': duration}
    else:
        return_data = {}
    return return_data

def get_image_size(image_path):
    # 打开图像文件
    with Image.open(image_path) as img:
        # 获取图像的宽度和高度
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
    #文件名根据年月日时分秒来命名
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
    # 排序文件名
    video_files.sort()
    return video_files

def save_image(image, path):
    tensor2pil(image).save(path)
    
def clear_memory():
    gc.collect()
    unload_all_models()
    soft_empty_cache()

def get_video_info_from_bytes(video_bytes):
    """
    从视频二进制数据中提取视频信息（fps、宽高、时长等）
    使用OpenCV和PyAV处理，避免磁盘持久化

    Args:
        video_bytes: 视频的二进制数据

    Returns:
        dict: 包含 fps, width, height, duration 的字典
    """
    try:
        # 创建临时文件用于OpenCV读取
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            # 使用OpenCV获取视频信息
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
            # 删除临时文件
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to get video info from bytes: {e}")

def extract_frames_from_bytes(video_bytes, output_dir, fps_scale=None):
    """
    从视频二进制数据中提取帧到指定目录（使用OpenCV）

    Args:
        video_bytes: 视频的二进制数据
        output_dir: 输出帧的目录
        fps_scale: 缩放因子 (可选)

    Returns:
        tuple: (帧数, fps, 宽度, 高度)
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            os.makedirs(output_dir, exist_ok=True)

            # 使用OpenCV打开视频
            cap = cv2.VideoCapture(tmp_path)

            if not cap.isOpened():
                raise ValueError("Unable to open video file")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 计算缩放后的尺寸
            if fps_scale and fps_scale > 0 and fps_scale != 1.0:
                scaled_width = int(width * fps_scale)
                scaled_height = int(height * fps_scale)
            else:
                scaled_width = width
                scaled_height = height

            # 逐帧读取并保存
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 缩放帧（如果需要）
                if fps_scale and fps_scale != 1.0:
                    frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 保存为PNG
                frame_path = os.path.join(output_dir, f'frame_{frame_count:08d}.png')
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(frame_path)

                frame_count += 1

            cap.release()

            return frame_count, fps, scaled_width, scaled_height
        finally:
            # 删除临时文件
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to extract frames from bytes: {e}")

def pack_images_to_video_bytes(images, fps, output_format='mp4'):
    """
    将IMAGE张量打包为视频二进制数据（使用OpenCV）

    Args:
        images: PyTorch张量 shape (batch, height, width, channels)
        fps: 帧率
        output_format: 输出格式 ('mp4', 'avi', 等)

    Returns:
        bytes: 视频二进制数据
    """
    try:
        # 创建临时文件用于存储视频
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}') as tmp:
            output_file = tmp.name

        try:
            height = images.shape[1]
            width = images.shape[2]

            # 选择编码器
            if output_format == 'mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif output_format == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            elif output_format == 'mov':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # 创建视频写入器
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            if not out.isOpened():
                raise ValueError(f"Failed to create video writer for format: {output_format}")

            # 逐帧写入
            for i in range(images.shape[0]):
                # 获取单个帧
                frame_tensor = images[i:i+1]  # shape: (1, height, width, channels)

                # 转换为PIL Image，然后转为numpy数组
                frame_pil = tensor2pil(frame_tensor)
                frame_np = np.array(frame_pil)  # RGB格式

                # 转换RGB到BGR（OpenCV使用BGR）
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                # 写入帧
                out.write(frame_bgr)

            out.release()

            # 读取生成的视频文件为字节
            with open(output_file, 'rb') as f:
                video_bytes = f.read()

            return video_bytes
        finally:
            # 删除临时文件
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to pack images to video bytes: {e}")

def extract_audio_from_bytes(video_bytes, audio_format='mp3'):
    """
    从视频二进制数据中提取音频（使用PyAV）

    Args:
        video_bytes: 视频的二进制数据
        audio_format: 音频格式 ('mp3', 'aac', 'wav', 等)

    Returns:
        bytes: 音频二进制数据
    """
    try:
        # 创建临时输入视频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        # 创建临时输出音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}') as tmp_audio:
            tmp_audio_path = tmp_audio.name

        try:
            # 使用PyAV打开视频并提取音频
            input_container = av.open(tmp_video_path)

            if not input_container.audio:
                raise ValueError("Video has no audio stream")

            # 确定输出格式和编码器
            if audio_format == 'mp3':
                codec_name = 'libmp3lame'
            elif audio_format == 'aac':
                codec_name = 'aac'
            elif audio_format == 'wav':
                codec_name = 'pcm_s16le'
            else:
                codec_name = 'libmp3lame'  # 默认为mp3

            # 创建输出容器
            output_container = av.open(tmp_audio_path, 'w')

            # 添加音频流
            audio_stream = input_container.audio[0]
            out_stream = output_container.add_stream(codec_name, rate=audio_stream.sample_rate)

            # 复制音频帧
            for frame in input_container.decode(audio=0):
                for packet in out_stream.encode(frame):
                    output_container.mux(packet)

            # 刷新缓冲区
            for packet in out_stream.encode():
                output_container.mux(packet)

            input_container.close()
            output_container.close()

            # 读取音频文件为字节
            with open(tmp_audio_path, 'rb') as f:
                audio_bytes = f.read()

            return audio_bytes
        finally:
            # 清理临时文件
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
    """
    给视频二进制数据添加图片水印（使用OpenCV）

    Args:
        video_bytes: 视频的二进制数据
        watermark_image: 水印图片（PIL Image对象或路径）
        watermark_width: 水印宽度
        position_x: 水印X位置
        position_y: 水印Y位置

    Returns:
        bytes: 添加水印后的视频二进制数据
    """
    try:
        # 处理水印图片，转换为numpy数组
        if isinstance(watermark_image, str):
            if not os.path.exists(watermark_image):
                raise ValueError(f"Watermark image not found: {watermark_image}")
            watermark_pil = Image.open(watermark_image).convert('RGBA')
        elif isinstance(watermark_image, Image.Image):
            watermark_pil = watermark_image.convert('RGBA')
        else:
            raise ValueError("Watermark image must be a file path or PIL Image")

        # 创建临时输入视频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        # 创建临时输出视频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
            output_path = tmp_output.name

        try:
            # 打开输入视频
            cap = cv2.VideoCapture(tmp_video_path)

            if not cap.isOpened():
                raise ValueError("Unable to open video file")

            # 获取视频参数
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 计算水印尺寸（按比例缩放）
            wm_pil = watermark_pil
            wm_w, wm_h = wm_pil.size
            wm_h_scaled = int(wm_h * watermark_width / wm_w)
            wm_pil = wm_pil.resize((watermark_width, wm_h_scaled), Image.Resampling.LANCZOS)

            # 转换为numpy数组
            watermark_np = np.array(wm_pil)

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise ValueError("Failed to create video writer")

            # 逐帧处理和写入
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 添加水印
                frame = add_watermark_to_frame(frame, watermark_np, position_x, position_y)

                # 写入帧
                out.write(frame)

            cap.release()
            out.release()

            # 读取输出视频文件为字节
            with open(output_path, 'rb') as f:
                output_bytes = f.read()

            return output_bytes
        finally:
            # 清理临时文件
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except:
                    pass
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to add watermark to video: {e}")

def get_video_bytes_from_input(video_input):
    """
    兼容处理视频输入 - 支持文件路径字符串或VideoData对象

    Args:
        video_input: 文件路径字符串或VideoData对象

    Returns:
        bytes: 视频二进制数据
    """
    from .video_types import VideoData

    if isinstance(video_input, VideoData):
        return video_input.to_bytes()
    elif isinstance(video_input, str):
        # 如果是文件路径，读取文件内容
        if not os.path.exists(video_input):
            raise ValueError(f"Video file not found: {video_input}")
        with open(video_input, 'rb') as f:
            return f.read()
    else:
        raise ValueError("Video input must be a file path (STRING) or VideoData object")

def get_audio_bytes_from_input(audio_input):
    """
    兼容处理音频输入 - 支持文件路径字符串或AudioData对象

    Args:
        audio_input: 文件路径字符串或AudioData对象

    Returns:
        bytes: 音频二进制数据
    """
    from .video_types import AudioData

    if isinstance(audio_input, AudioData):
        return audio_input.to_bytes()
    elif isinstance(audio_input, str):
        # 如果是文件路径，读取文件内容
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        with open(audio_input, 'rb') as f:
            return f.read()
    else:
        raise ValueError("Audio input must be a file path (STRING) or AudioData object")

def add_watermark_to_frame(frame, watermark, x, y):
    """
    给单个视频帧添加水印

    Args:
        frame: OpenCV格式的视频帧 (BGR)
        watermark: numpy数组格式的水印图片 (RGBA)
        x: 水印X位置
        y: 水印Y位置

    Returns:
        添加水印后的帧
    """
    frame_height, frame_width = frame.shape[:2]
    wm_height, wm_width = watermark.shape[:2]

    # 限制水印位置在有效范围内
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    # 计算水印区域
    x_end = min(x + wm_width, frame_width)
    y_end = min(y + wm_height, frame_height)

    # 调整水印大小以适应边界
    wm_w_adjusted = x_end - x
    wm_h_adjusted = y_end - y

    if wm_w_adjusted <= 0 or wm_h_adjusted <= 0:
        return frame

    watermark_region = watermark[:wm_h_adjusted, :wm_w_adjusted]

    # 分离RGB和Alpha通道
    if watermark_region.shape[2] == 4:
        watermark_rgb = watermark_region[:, :, :3]
        watermark_alpha = watermark_region[:, :, 3] / 255.0
    else:
        watermark_rgb = watermark_region
        watermark_alpha = np.ones((wm_h_adjusted, wm_w_adjusted))

    # 转换RGB到BGR
    watermark_bgr = cv2.cvtColor(watermark_rgb, cv2.COLOR_RGB2BGR)

    # 融合水印到帧
    frame_roi = frame[y:y_end, x:x_end]

    for c in range(3):
        frame_roi[:, :, c] = frame_roi[:, :, c] * (1 - watermark_alpha) + watermark_bgr[:, :, c] * watermark_alpha

    frame[y:y_end, x:x_end] = frame_roi

    return frame