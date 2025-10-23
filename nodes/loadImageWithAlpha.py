import torch


class CombineImageWithAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgba",)
    FUNCTION = "combine"
    OUTPUT_NODE = False
    CATEGORY = "🔥FFmpeg/Image"

    def combine(self, image, mask):
        try:
            print(f"[CombineImageWithAlpha] Input image shape: {image.shape}")
            print(f"[CombineImageWithAlpha] Input mask shape: {mask.shape}")

            if len(mask.shape) == 4:
                mask = mask.squeeze(1)

            alpha = 1.0 - mask
            alpha = alpha.unsqueeze(-1)
            image_rgba = torch.cat([image, alpha], dim=-1)

            print(f"[CombineImageWithAlpha] Output RGBA shape: {image_rgba.shape}, channels: {image_rgba.shape[3]}")

            return (image_rgba,)

        except Exception as e:
            raise ValueError(f"Failed to combine image with alpha: {e}")


NODE_CLASS_MAPPINGS = {
    "CombineImageWithAlpha": CombineImageWithAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineImageWithAlpha": "Combine Image With Alpha (合并透明度)",
}
