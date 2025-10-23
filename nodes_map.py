from .nodes.addTextWatermark import *
from .nodes.frames2video import *
from .nodes.video2frames import *
from .nodes.addImgWatermark import *
from .nodes.videoFlip import *
from .nodes.extractAudio import *
from .nodes.loadImageFromDir import *
from .nodes.imageCopy import *
from .nodes.imagePath2Tensor import *
from .nodes.mergingVideoByTwo import *
from .nodes.mergingVideoByPlenty import *
from .nodes.stitchingVideo import *
from .nodes.multiCuttingVideo import *
from .nodes.singleCuttingVideo import *
from .nodes.addAudio import *
from .nodes.imagesSave import *
from .nodes.pipVideo import *
from .nodes.videoTransition import *
from .nodes.videoPlayback import *
from .nodes.getVideoBase64 import *
from .nodes.videoToBase64 import *
from .nodes.getVideoInfo import *
from .nodes.addWatermarkToVideo import *
from .nodes.packVideo import *
from .nodes.imageToVideoAPI import *
from .nodes.loadImageWithAlpha import *
from .nodes.loadBase64Image import *

NODE_CLASS_MAPPINGS = {
    "Video2Frames": Video2Frames,
    "Frames2Video": Frames2Video,
    "AddTextWatermark": AddTextWatermark,
    "AddImgWatermark": AddImgWatermark,
    "VideoFlip": VideoFlip,
    "ExtractAudio": ExtractAudio,
    "LoadImageFromDir": LoadImageFromDir,
    "ImageCopy": ImageCopy,
    "ImagePath2Tensor": ImagePath2Tensor,
    "MergingVideoByTwo": MergingVideoByTwo,
    "MergingVideoByPlenty": MergingVideoByPlenty,
    "StitchingVideo": StitchingVideo,
    "MultiCuttingVideo": MultiCuttingVideo,
    "SingleCuttingVideo": SingleCuttingVideo,
    "AddAudio": AddAudio,
    "ImagesSave": ImagesSave,
    "PipVideo": PipVideo,
    "VideoTransition": VideoTransition,
    "VideoPlayback": VideoPlayback,
    "GetVideoBase64": GetVideoBase64,
    "VideoToBase64": VideoToBase64,
    "GetVideoInfo": GetVideoInfo,
    "AddWatermarkToVideo": AddWatermarkToVideo,
    "PackVideo": PackVideo,
    "ImageToVideoAPI": ImageToVideoAPI,
    "CombineImageWithAlpha": CombineImageWithAlpha,
    "LoadBase64Image": LoadBase64Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Video2Frames": "🔥Video2Frames",
    "Frames2Video": "🔥Frames2Video",
    "AddTextWatermark": "🔥AddTextWatermark",
    "AddImgWatermark": "🔥AddImgWatermark",
    "VideoFlip": "🔥VideoFlip",
    "ExtractAudio": "🔥ExtractAudio",
    "LoadImageFromDir": "🔥LoadImageFromDir",
    "ImageCopy": "🔥ImageCopy",
    "ImagePath2Tensor": "🔥ImagePath2Tensor",
    "MergingVideoByTwo": "🔥MergingVideoByTwo",
    "MergingVideoByPlenty": "🔥MergingVideoByPlenty",
    "StitchingVideo": "🔥StitchingVideo",
    "MultiCuttingVideo": "🔥MultiCuttingVideo",
    "SingleCuttingVideo": "🔥SingleCuttingVideo",
    "AddAudio": "🔥AddAudio",
    "ImagesSave": "🔥ImagesSave",
    "PipVideo": "🔥PipVideo",
    "VideoTransition": "🔥VideoTransition",
    "VideoPlayback": "🔥VideoPlayback",
    "GetVideoBase64": "🔥GetVideoBase64",
    "VideoToBase64": "🔥VideoToBase64",
    "GetVideoInfo": "🔥GetVideoInfo",
    "AddWatermarkToVideo": "🔥AddWatermarkToVideo",
    "PackVideo": "🔥PackVideo",
    "ImageToVideoAPI": "🔥ImageToVideoAPI",
    "CombineImageWithAlpha": "🔥Combine Image With Alpha",
    "LoadBase64Image": "🔥Load Base64 Image",
}
