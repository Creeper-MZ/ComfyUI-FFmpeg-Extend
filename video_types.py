"""
自定义VIDEO类型，用于在节点间传递video数据（不涉及文件IO）
支持base64编码用于网络传输
"""

import base64
import io
import json
from typing import Optional, Dict, Any


class AnyType(str):
    """通用类型匹配器 - 可以接受任何类型"""
    def __ne__(self, __value: object) -> bool:
        return False


# 自定义类型定义
video_type = AnyType("VIDEO")
audio_type = AnyType("AUDIO")
video_or_string = AnyType("VIDEO_OR_STRING")  # 同时接受VIDEO和STRING
audio_or_string = AnyType("AUDIO_OR_STRING")  # 同时接受AUDIO和STRING


class VideoData:
    """
    VIDEO类型的数据结构
    内存中存储video的二进制数据和元信息
    """

    def __init__(self, video_bytes: bytes, metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            video_bytes: 视频文件的二进制数据
            metadata: 视频元信息 (可选)，包含如:
                {
                    'fps': 30.0,
                    'width': 1920,
                    'height': 1080,
                    'duration': 10.5,
                    'format': 'mp4'
                }
        """
        self.video_bytes = video_bytes
        self.metadata = metadata or {}

    def to_base64(self) -> str:
        """将video转换为base64字符串"""
        return base64.b64encode(self.video_bytes).decode('utf-8')

    @classmethod
    def from_base64(cls, base64_str: str, metadata: Optional[Dict[str, Any]] = None) -> 'VideoData':
        """从base64字符串创建VideoData"""
        video_bytes = base64.b64decode(base64_str)
        return cls(video_bytes, metadata)

    def to_bytes(self) -> bytes:
        """获取原始字节数据"""
        return self.video_bytes

    def get_metadata(self) -> Dict[str, Any]:
        """获取元信息"""
        return self.metadata.copy()

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """更新元信息"""
        self.metadata.update(metadata)

    def get_size(self) -> int:
        """获取video大小（字节）"""
        return len(self.video_bytes)

    def __repr__(self):
        size_mb = self.get_size() / (1024 * 1024)
        return f"VideoData(size={size_mb:.2f}MB, metadata={self.metadata})"


class AudioData:
    """
    AUDIO类型的数据结构
    内存中存储audio的二进制数据和元信息
    """

    def __init__(self, audio_bytes: bytes, metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            audio_bytes: 音频文件的二进制数据
            metadata: 音频元信息 (可选)，包含如:
                {
                    'sample_rate': 48000,
                    'channels': 2,
                    'duration': 10.5,
                    'format': 'aac'
                }
        """
        self.audio_bytes = audio_bytes
        self.metadata = metadata or {}

    def to_base64(self) -> str:
        """将audio转换为base64字符串"""
        return base64.b64encode(self.audio_bytes).decode('utf-8')

    @classmethod
    def from_base64(cls, base64_str: str, metadata: Optional[Dict[str, Any]] = None) -> 'AudioData':
        """从base64字符串创建AudioData"""
        audio_bytes = base64.b64decode(base64_str)
        return cls(audio_bytes, metadata)

    def to_bytes(self) -> bytes:
        """获取原始字节数据"""
        return self.audio_bytes

    def get_metadata(self) -> Dict[str, Any]:
        """获取元信息"""
        return self.metadata.copy()

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """更新元信息"""
        self.metadata.update(metadata)

    def get_size(self) -> int:
        """获取audio大小（字节）"""
        return len(self.audio_bytes)

    def __repr__(self):
        size_mb = self.get_size() / (1024 * 1024)
        return f"AudioData(size={size_mb:.2f}MB, metadata={self.metadata})"
