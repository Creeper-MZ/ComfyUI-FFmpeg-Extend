import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import base64
import io
import node_helpers


class LoadBase64Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_string": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Image"

    def load_image(self, base64_string):
        try:
            base64_string = base64_string.strip()

            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',', 1)[1]

            image_data = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(image_data))

            output_images = []
            output_masks = []
            w, h = None, None

            excluded_formats = ['MPO']

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]

                if image.size[0] != w or image.size[1] != h:
                    continue

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]

                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                elif i.mode == 'P' and 'transparency' in i.info:
                    mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            return (output_image, output_mask)

        except Exception as e:
            raise ValueError(f"Failed to load base64 image: {e}")


NODE_CLASS_MAPPINGS = {
    "LoadBase64Image": LoadBase64Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBase64Image": "🔥Load Base64 Image",
}
