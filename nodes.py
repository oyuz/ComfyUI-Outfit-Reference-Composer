import json
import numpy as np
import torch
from PIL import Image, ImageFilter


SLOTS = ("top", "bottom", "shoes", "hat", "bag", "glasses", "necklace", "earrings", "bracelet")
ANCHORS = {
    "female_front": {"top": (.50, .18, .43), "bottom": (.50, .43, .37), "shoes": (.50, .82, .34), "hat": (.50, .035, .27), "glasses": (.50, .135, .22), "bag": (.77, .55, .19), "necklace": (.50, .255, .22), "earrings": (.50, .145, .24), "bracelet": (.72, .61, .14)},
    "male_front": {"top": (.50, .17, .46), "bottom": (.50, .43, .39), "shoes": (.50, .82, .36), "hat": (.50, .03, .28), "glasses": (.50, .13, .23), "bag": (.78, .55, .20), "necklace": (.50, .245, .22), "earrings": (.50, .14, .24), "bracelet": (.72, .61, .14)},
}


class OutfitReferenceComposer:
    """Deterministically composes selected product images into a flat outfit reference."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "outfit_spec": ("STRING", {"multiline": True, "default": '{\n  "items": {}\n}'}),
            "avatar_profile": (["female_front", "male_front"], {"default": "female_front"}),
            "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            "background": (["off_white", "white", "light_gray"], {"default": "off_white"}),
        }
        return {"required": required, "optional": {slot: ("IMAGE",) for slot in SLOTS}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("outfit_reference", "outfit_mask", "outfit_prompt")
    FUNCTION = "compose"
    CATEGORY = "Outfit Reference"

    @staticmethod
    def _image(tensor):
        arr = tensor[0].detach().cpu().numpy()
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).convert("RGB")

    @staticmethod
    def _cutout(image):
        data = np.asarray(image).astype(np.int16)
        # MVP assumption: items are centered on a clean white / near-white background.
        alpha = np.where(np.min(data, axis=2) > 244, 0, 255).astype(np.uint8)
        bbox = Image.fromarray(alpha).getbbox()
        if not bbox:
            return None
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha))
        return rgba.crop(bbox)

    @staticmethod
    def _parse(spec):
        try:
            value = json.loads(spec)
        except json.JSONDecodeError as error:
            raise ValueError(f"outfit_spec is not valid JSON: {error.msg}") from error
        items = value.get("items", value)
        if not isinstance(items, dict):
            raise ValueError("outfit_spec.items must be an object keyed by slot")
        return items

    def compose(self, outfit_spec, avatar_profile, width, height, background, **images):
        items = self._parse(outfit_spec)
        colors = {"white": (255, 255, 255), "off_white": (248, 248, 245), "light_gray": (238, 238, 238)}
        canvas = Image.new("RGBA", (width, height), colors[background] + (255,))
        combined_mask = Image.new("L", (width, height), 0)
        anchors = ANCHORS[avatar_profile]

        # Garments first, then accessories. This keeps jewellery readable in the reference image.
        for slot in ("bottom", "top", "shoes", "hat", "bag", "necklace", "earrings", "glasses", "bracelet"):
            if slot not in items or images.get(slot) is None:
                continue
            item = items[slot] if isinstance(items[slot], dict) else {}
            layout = item.get("node_layout", item)
            cutout = self._cutout(self._image(images[slot]))
            if cutout is None:
                continue
            cx, y, relative_width = anchors[slot]
            fit = layout.get("fit", "regular")
            relative_width *= {"slim": .88, "regular": 1, "loose": 1.10, "oversized": 1.20}.get(fit, 1)
            target_width = max(1, int(width * relative_width))
            target_height = max(1, int(cutout.height * target_width / cutout.width))
            cutout = cutout.resize((target_width, target_height), Image.Resampling.LANCZOS)
            x = int(width * cx - target_width / 2)
            top = int(height * y)
            if slot == "shoes":
                top = min(top, height - target_height - int(height * .03))
            if slot == "bag":
                # Product decision: every bag is presented as a right-hand carry.
                x = int(width * .73)
            shadow = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
            shadow.putalpha(cutout.getchannel("A").filter(ImageFilter.GaussianBlur(3)).point(lambda p: p // 7))
            canvas.alpha_composite(shadow, (x + 3, top + 5))
            canvas.alpha_composite(cutout, (x, top))
            combined_mask.paste(cutout.getchannel("A"), (x, top), cutout.getchannel("A"))

        output = torch.from_numpy(np.asarray(canvas.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
        mask = torch.from_numpy(np.asarray(combined_mask).astype(np.float32) / 255.0).unsqueeze(0)
        prompt = "Full-body front-facing 3D cartoon character wearing the exact complete outfit and accessories shown in the outfit reference."
        return (output, mask, prompt)


NODE_CLASS_MAPPINGS = {"OutfitReferenceComposer": OutfitReferenceComposer}
NODE_DISPLAY_NAME_MAPPINGS = {"OutfitReferenceComposer": "Outfit Reference Composer"}
