import os
import subprocess
from PIL import Image
from ..func import get_image_size,set_file_name,video_type,add_watermark_to_video_bytes_pyav


class AddImgWatermark:
 
    # 初始化方法
    def __init__(self): 
        pass 
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default":"C:/Users/Desktop/video.mp4",}),
                "output_path": ("STRING", {"default":"C:/Users/Desktop/output/",}),
                "watermark_image": ("STRING", {"default":"C:/Users/Desktop/logo.png",}),
                "watermark_img_width":  ("INT", {"default": 100,"min": 1, "step": 1}),
                "position_x":  ("INT", {"default": 10, "step": 1}),
                "position_y":  ("INT", {"default": 10, "step": 1}),
                "use_pyav": ("BOOLEAN", {"default": True, "label_on": "PyAV (透明支持)", "label_off": "FFmpeg"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_complete_path",)
    FUNCTION = "add_img_watermark" 
    OUTPUT_NODE = True
    CATEGORY = "🔥FFmpeg" 

    def add_img_watermark(self,video_path,output_path,watermark_image,watermark_img_width,position_x,position_y,use_pyav=True):
        try:

            video_path = os.path.abspath(video_path).strip()
            output_path = os.path.abspath(output_path).strip()
            # 视频不存在
            if not video_path.lower().endswith(video_type()):
                raise ValueError("video_path："+video_path+"不是视频文件（video_path:"+video_path+" is not a video file）")

            if not os.path.exists(video_path):
                raise ValueError("video_path："+video_path+"不存在（video_path:"+video_path+" does not exist）")

            #判断output_path是否是一个目录
            if not os.path.isdir(output_path):
                raise ValueError("output_path："+output_path+"不是目录（output_path:"+output_path+" is not a directory）")

            # 文件不是图片
            if not watermark_image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                raise ValueError("watermark_image不是图片文件（watermark file is not a image file）")

            if not os.path.exists(watermark_image):
                raise ValueError("watermark_image："+watermark_image+"不存在（watermark_image :"+watermark_image+" does not exist）")

            file_name = set_file_name(video_path)
            output_file_path = os.path.join(output_path, file_name)

            if use_pyav:
                # 使用PyAV方法（完全支持透明PNG）
                print("[AddImgWatermark] Using PyAV for transparent watermark support")

                # 读取视频文件
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()

                # 使用PyAV添加水印
                output_bytes = add_watermark_to_video_bytes_pyav(
                    video_bytes,
                    watermark_image,
                    watermark_img_width,
                    position_x,
                    position_y
                )

                # 写入输出文件
                with open(output_file_path, 'wb') as f:
                    f.write(output_bytes)

                print(f"[AddImgWatermark] PyAV watermark completed: {output_file_path}")
            else:
                # 使用FFmpeg方法（旧版本，透明度支持可能有问题）
                print("[AddImgWatermark] Using FFmpeg")
                width,height = get_image_size(watermark_image)
                watermark_img_height = int(height * watermark_img_width / width)  # 按比例计算新高度
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', watermark_image,
                    '-filter_complex',f"[1:v]scale={watermark_img_width}:{watermark_img_height}[wm];[0:v][wm]overlay={position_x}:{position_y}",
                    '-c:v', 'libx264',  # H.264 编码器
                    '-crf', '0',  # 视觉无损质量（0 = 最高质量）
                    '-preset', 'slow',  # 慢速编码，更好的压缩率
                    '-c:a', 'copy',  # 音频直接复制，避免重新编码
                    output_file_path,
                ]
                result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                # 检查返回码
                if result.returncode != 0:
                    # 如果有错误，输出错误信息
                     print(f"Error: {result.stderr.decode('utf-8')}")
                     raise ValueError(f"Error: {result.stderr.decode('utf-8')}")
                else:
                    # 输出标准输出信息
                    print(result.stdout)
        except Exception as e:
            raise ValueError(e)
        return (output_file_path,)