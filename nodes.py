import json
from collections import deque

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


SLOTS = (
    "top",
    "bottom",
    "socks",
    "shoes",
    "hat",
    "bag",
    "glasses",
    "necklace",
    "earrings",
    "bracelet",
)

DRAW_ORDER = (
    "bottom",
    "socks",
    "shoes",
    "top",
    "hat",
    "earrings",
    "glasses",
    "bag",
    "bracelet",
    "necklace",
)

# All coordinates are normalized to the output canvas. The main outfit is
# slightly left of centre so the bag and wrist accessory have a dedicated
# gutter on the right. These are maximum envelopes; top, bottom and socks
# return metadata-specific hard boxes whose occupied ranges do not overlap.
TEMPLATE = {
    "body_center_x": 0.45,
    "hard_boxes": {
        "top": (0.14, 0.17, 0.73, 0.51),
        "bottom": (0.20, 0.53, 0.70, 0.88),
        "socks": (0.28, 0.74, 0.62, 0.885),
        "shoes": (0.27, 0.90, 0.63, 0.98),
        "hat": (0.33, 0.015, 0.57, 0.10),
        "glasses": (0.35, 0.105, 0.55, 0.15),
        # A paired earring image is split and placed on the two sides of the
        # glasses. The three boxes are disjoint.
        "earrings_left": (0.25, 0.105, 0.34, 0.165),
        "earrings_right": (0.56, 0.105, 0.65, 0.165),
        "necklace": (0.34, 0.19, 0.56, 0.31),
        "bracelet": (0.77, 0.31, 0.94, 0.43),
        "bag": (0.75, 0.47, 0.96, 0.75),
    },
    "landmarks": {
        "neck_y": 0.19,
        "shoulder_y": 0.23,
        "top_hem_y": {
            "crop": 0.40,
            "waist": 0.455,
            "hip": 0.49,
            "upper_thigh": 0.505,
        },
        "bottom_waist_y": {"high": 0.53, "mid": 0.55, "low": 0.57},
        "bottom_hem_y": {
            "thigh": 0.67,
            "knee": 0.73,
            "calf": 0.81,
            "ankle": 0.86,
            "floor": 0.88,
        },
        # These are shoulder spans, not full sleeve-to-sleeve widths.
        "shoulder_width": {
            "slim": 0.20,
            "regular": 0.225,
            "loose": 0.255,
            "oversized": 0.29,
        },
    },
}

SLEEVE_SHOULDER_RATIO = {
    "sleeveless": 0.84,
    "cap": 0.76,
    "short": 0.68,
    "elbow": 0.60,
    "long": 0.65,
}

TOP_TYPE_SHOULDER_RATIO = {
    "tank": 0.84,
    "camisole": 0.88,
    "jersey": 0.68,
    "tee": 0.67,
    "shirt": 0.65,
    "hoodie": 0.62,
    "jacket": 0.62,
    "outerwear": 0.62,
    "layered_outerwear": 0.62,
}


class OutfitReferenceComposer:
    """Compose complete product shots in a deterministic adult outfit template."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "outfit_spec": ("STRING", {"multiline": True, "default": '{\n  "items": {}\n}'}),
            "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            "background": (["off_white", "white", "light_gray"], {"default": "off_white"}),
            "show_layout_guides": ("BOOLEAN", {"default": False}),
        }
        return {"required": required, "optional": {slot: ("IMAGE",) for slot in SLOTS}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("outfit_reference",)
    FUNCTION = "compose"
    CATEGORY = "Outfit Reference"

    @staticmethod
    def _image(tensor):
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(
                "each outfit slot accepts exactly one IMAGE; split image batches before this node"
            )
        arr = tensor[0].detach().cpu().numpy()
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            return Image.fromarray(arr, "RGBA")
        return Image.fromarray(arr[..., :3], "RGB").convert("RGBA")

    @staticmethod
    def _components(binary):
        """Return connected components for a small boolean mask."""
        height, width = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        components = []

        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                queue = deque([(x, y)])
                visited[y, x] = True
                pixels = []
                min_x = max_x = x
                min_y = max_y = y
                while queue:
                    px, py = queue.popleft()
                    pixels.append((px, py))
                    min_x = min(min_x, px)
                    max_x = max(max_x, px)
                    min_y = min(min_y, py)
                    max_y = max(max_y, py)
                    for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                        if 0 <= nx < width and 0 <= ny < height and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
                components.append(
                    {
                        "pixels": pixels,
                        "area": len(pixels),
                        "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                    }
                )
        return components

    @classmethod
    def _component_keep_mask(cls, binary, slot):
        """Keep the product while rejecting disconnected watermarks and dust."""
        height, width = binary.shape
        max_side = 256
        scale = min(1.0, max_side / max(width, height))
        small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        small = Image.fromarray((binary * 255).astype(np.uint8), "L").resize(
            small_size, Image.Resampling.NEAREST
        )
        small = small.filter(ImageFilter.MaxFilter(3))
        small_binary = np.asarray(small) > 0
        components = cls._components(small_binary)
        if not components:
            return np.ones((height, width), dtype=np.uint8) * 255

        components.sort(key=lambda component: component["area"], reverse=True)
        largest = components[0]["area"]
        pair_like = slot in {"socks", "shoes", "glasses", "earrings", "bracelet"}
        max_components = 8 if pair_like else (12 if slot == "necklace" else 5)
        selected = []
        for component in components:
            if len(selected) >= max_components:
                break
            relative_area = component["area"] / largest
            x0, y0, x1, y1 = component["bbox"]
            centre_x = (x0 + x1) / (2 * small_size[0])
            centre_y = (y0 + y1) / (2 * small_size[1])
            is_central = 0.08 <= centre_x <= 0.92 and 0.05 <= centre_y <= 0.95
            if not selected:
                selected.append(component)
            elif relative_area >= 0.018 and (is_central or relative_area >= 0.18):
                selected.append(component)

        keep_small = np.zeros((small_size[1], small_size[0]), dtype=np.uint8)
        for component in selected:
            for x, y in component["pixels"]:
                keep_small[y, x] = 255
        keep_small = np.asarray(
            Image.fromarray(keep_small, "L").filter(ImageFilter.MaxFilter(5))
        )
        return np.asarray(
            Image.fromarray(keep_small, "L").resize((width, height), Image.Resampling.NEAREST)
        )

    @classmethod
    def _cutout(cls, image, slot, layout):
        """Create a tight product cutout from alpha or a light studio background."""
        source_alpha = np.asarray(image.getchannel("A"))
        if source_alpha.min() < 250:
            alpha = source_alpha
        else:
            rgb = np.asarray(image.convert("RGB")).astype(np.int16)
            height, width = rgb.shape[:2]
            border_size = max(2, int(min(width, height) * 0.015))
            border = np.concatenate(
                (
                    rgb[:border_size].reshape(-1, 3),
                    rgb[-border_size:].reshape(-1, 3),
                    rgb[:, :border_size].reshape(-1, 3),
                    rgb[:, -border_size:].reshape(-1, 3),
                ),
                axis=0,
            )
            background_rgb = np.median(border, axis=0)
            distance = np.max(np.abs(rgb - background_rgb), axis=2).astype(np.float32)
            border_distance = np.max(np.abs(border - background_rgb), axis=1)
            automatic_threshold = float(np.percentile(border_distance, 90)) + 10.0
            try:
                threshold = float(
                    layout.get("background_threshold", np.clip(automatic_threshold, 18, 34))
                )
            except (TypeError, ValueError) as error:
                raise ValueError("background_threshold must be a number from 4 to 100") from error
            if not 4 <= threshold <= 100:
                raise ValueError("background_threshold must be a number from 4 to 100")
            binary = distance >= threshold
            keep = cls._component_keep_mask(binary, slot).astype(np.float32) / 255.0
            lower = max(4.0, threshold * 0.45)
            upper = max(lower + 1.0, threshold * 1.35)
            soft_alpha = np.clip((distance - lower) / (upper - lower), 0.0, 1.0)
            alpha = np.round(soft_alpha * keep * 255.0).astype(np.uint8)

        alpha_image = Image.fromarray(alpha, "L")
        bbox = alpha_image.point(lambda value: 255 if value >= 8 else 0).getbbox()
        if not bbox:
            return None
        pad = max(2, int(min(image.size) * 0.002))
        bbox = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(image.width, bbox[2] + pad),
            min(image.height, bbox[3] + pad),
        )
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha_image)
        return rgba.crop(bbox)

    @staticmethod
    def _parse(spec):
        try:
            value = json.loads(spec)
        except json.JSONDecodeError as error:
            raise ValueError(f"outfit_spec is not valid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError("outfit_spec root must be a JSON object")
        items = value.get("items", value)
        if not isinstance(items, dict):
            raise ValueError("outfit_spec.items must be an object keyed by slot")
        return items

    @staticmethod
    def _layout(item):
        if item is None:
            return {}
        if not isinstance(item, dict):
            raise ValueError("each outfit_spec item must be a JSON object")
        nested = item.get("node_layout")
        if isinstance(nested, dict):
            return {**item, **nested}
        return item

    @staticmethod
    def _choice(layout, key, default, allowed, aliases=None):
        raw = layout.get(key, default)
        if not isinstance(raw, str):
            raise ValueError(f"{key} must be a string")
        value = raw.strip().lower()
        if aliases:
            value = aliases.get(value, value)
        if value not in allowed:
            options = ", ".join(sorted(option for option in allowed if option))
            item_id = layout.get("item_id", "item")
            raise ValueError(f"{item_id}.{key} must be one of: {options}; got {raw!r}")
        return value

    @staticmethod
    def _normal_box_to_pixels(box, width, height):
        x0, y0, x1, y1 = box
        return (
            int(round(x0 * width)),
            int(round(y0 * height)),
            int(round(x1 * width)),
            int(round(y1 * height)),
        )

    @staticmethod
    def _bounded_size(
        cutout,
        target_width,
        target_height,
        hard_width,
        hard_height,
        max_axis_distortion=2.5,
    ):
        """Apply semantic width and height independently inside a hard slot.

        Product-shot aspect ratios are not reliable body proportions: a long-
        sleeve shirt naturally occupies a taller/narrower source box than a
        short-sleeve crop top. Preserving that source aspect ratio made the
        shoulder anchor drift. Layout metadata therefore owns both axes; the
        source image only supplies the pixels inside those anchors.
        """
        width = max(1.0, min(float(target_width), float(hard_width)))
        height = max(1.0, min(float(target_height), float(hard_height)))
        source_aspect = cutout.width / max(1.0, float(cutout.height))
        target_aspect = width / height
        axis_distortion = max(target_aspect / source_aspect, source_aspect / target_aspect)
        if axis_distortion > max_axis_distortion:
            raise ValueError(
                f"anchor normalization would require {axis_distortion:.2f}x axis distortion "
                f"(limit {max_axis_distortion:.2f}x); check the cutout/metadata or raise "
                "max_axis_distortion for this item"
            )
        return int(round(width)), int(round(height))

    @staticmethod
    def _max_axis_distortion(layout):
        try:
            value = float(layout.get("max_axis_distortion", 2.5))
        except (TypeError, ValueError) as error:
            raise ValueError("max_axis_distortion must be a number from 1.0 to 4.0") from error
        if not 1.0 <= value <= 4.0:
            raise ValueError("max_axis_distortion must be a number from 1.0 to 4.0")
        return value

    def _top_placement(self, cutout, layout, width, height):
        landmarks = TEMPLATE["landmarks"]
        fit = self._choice(
            layout,
            "fit",
            "regular",
            set(landmarks["shoulder_width"]),
            {"oversize": "oversized"},
        )
        length = self._choice(layout, "length", "waist", set(landmarks["top_hem_y"]))
        coverage = self._choice(
            layout,
            "coverage",
            "",
            {"", "show_midriff", "cover_waist", "cover_hip"},
        )
        if coverage == "show_midriff":
            length = "crop"
        elif coverage == "cover_waist" and length == "crop":
            length = "waist"
        elif coverage == "cover_hip" and length in {"crop", "waist"}:
            length = "hip"
        hem_y = landmarks["top_hem_y"].get(length, landmarks["top_hem_y"]["waist"])
        hard_normal = list(TEMPLATE["hard_boxes"]["top"])
        hard_normal[3] = min(hard_normal[3], hem_y)
        hard = self._normal_box_to_pixels(tuple(hard_normal), width, height)
        hard_width, hard_height = hard[2] - hard[0], hard[3] - hard[1]

        garment_type = self._choice(
            layout,
            "garment_type",
            str(layout.get("silhouette", "")),
            {"", *TOP_TYPE_SHOULDER_RATIO},
            {
                "vest": "tank",
                "t_shirt": "tee",
                "t-shirt": "tee",
                "jacket_with_inner": "layered_outerwear",
            },
        )
        sleeve = self._choice(layout, "sleeve_length", "", {"", *SLEEVE_SHOULDER_RATIO})
        shoulder_ratio = TOP_TYPE_SHOULDER_RATIO.get(garment_type)
        if shoulder_ratio is None:
            shoulder_ratio = SLEEVE_SHOULDER_RATIO.get(sleeve, 0.64)
        try:
            shoulder_ratio = float(layout.get("source_shoulder_ratio", shoulder_ratio))
        except (TypeError, ValueError) as error:
            raise ValueError("source_shoulder_ratio must be a number from 0.30 to 1.0") from error
        if not 0.30 <= shoulder_ratio <= 1.0:
            raise ValueError("source_shoulder_ratio must be a number from 0.30 to 1.0")
        target_shoulder = landmarks["shoulder_width"].get(
            fit, landmarks["shoulder_width"]["regular"]
        ) * width
        target_width = target_shoulder / shoulder_ratio
        desired_height = max(1.0, (hem_y - landmarks["neck_y"]) * height)
        target_width, target_height = self._bounded_size(
            cutout,
            target_width,
            desired_height,
            hard_width,
            hard_height,
            self._max_axis_distortion(layout),
        )
        centre_x = TEMPLATE["body_center_x"] * width
        x = int(round(centre_x - target_width / 2))
        y = int(round(hem_y * height - target_height))
        x = min(max(x, hard[0]), hard[2] - target_width)
        y = min(max(y, hard[1]), hard[3] - target_height)
        label = f"top {fit}/{length}"
        return (x, y, target_width, target_height), hard, label

    def _bottom_placement(self, cutout, layout, width, height):
        landmarks = TEMPLATE["landmarks"]
        fit = self._choice(
            layout,
            "fit",
            "regular",
            {"slim", "regular", "loose", "oversized"},
            {"oversize": "oversized"},
        )
        self._choice(layout, "garment_type", "", {"", "shorts", "pants", "skirt"})
        waist = self._choice(layout, "waist", "mid", set(landmarks["bottom_waist_y"]))
        length = self._choice(
            layout,
            "length",
            "floor",
            set(landmarks["bottom_hem_y"]),
            {"full": "floor"},
        )
        silhouette_values = {
            "slim",
            "straight",
            "wide",
            "baggy",
            "flare",
            "skirt",
            "pleated",
            "a_line",
            "voluminous",
        }
        silhouette = self._choice(
            layout,
            "silhouette",
            "straight",
            silhouette_values,
            {
                "wide_pants": "wide",
                "straight_pants": "straight",
                "a-line": "a_line",
            },
        )
        waist_y = landmarks["bottom_waist_y"].get(waist, landmarks["bottom_waist_y"]["mid"])
        hem_y = landmarks["bottom_hem_y"].get(length, landmarks["bottom_hem_y"]["floor"])
        hard_normal = list(TEMPLATE["hard_boxes"]["bottom"])
        hard_normal[3] = min(hard_normal[3], hem_y)
        hard = self._normal_box_to_pixels(tuple(hard_normal), width, height)
        hard_width, hard_height = hard[2] - hard[0], hard[3] - hard[1]
        desired_height = max(1.0, (hem_y - waist_y) * height)

        fit_width = {"slim": 0.27, "regular": 0.31, "loose": 0.36, "oversized": 0.41}.get(
            fit, 0.31
        )
        silhouette_multiplier = {
            "slim": 0.88,
            "straight": 0.92,
            "wide": 1.00,
            "baggy": 1.08,
            "flare": 1.03,
            "skirt": 0.94,
            "pleated": 0.98,
            "a_line": 1.00,
            "voluminous": 1.08,
        }.get(silhouette, 1.0)
        desired_width = fit_width * silhouette_multiplier * width
        # Product-shot aspect ratios are not body measurements. Waist/hem own
        # the vertical axis, while fit/silhouette own the horizontal axis.
        # This gives two products with the same metadata the same body scale.
        target_height = max(1, min(int(round(desired_height)), hard_height))
        target_width = max(1, min(int(round(desired_width)), hard_width))
        self._bounded_size(
            cutout,
            target_width,
            target_height,
            hard_width,
            hard_height,
            self._max_axis_distortion(layout),
        )
        centre_x = TEMPLATE["body_center_x"] * width
        x = int(round(centre_x - target_width / 2))
        y = int(round(waist_y * height))
        x = min(max(x, hard[0]), hard[2] - target_width)
        y = min(max(y, hard[1]), hard[3] - target_height)
        label = f"bottom {waist}/{length}/{silhouette}"
        return (x, y, target_width, target_height), hard, label

    def _socks_placement(self, cutout, layout, bottom_layout, width, height, bottom_present=False):
        bottom_length = self._choice(
            bottom_layout,
            "length",
            "" if not bottom_present else "floor",
            {"", *TEMPLATE["landmarks"]["bottom_hem_y"]},
            {"full": "floor"},
        )
        if bottom_present and not bottom_length:
            bottom_length = "floor"
        if bottom_present and bottom_length not in {"thigh", "knee"}:
            raise ValueError(
                "socks cannot share the leg region with a calf/ankle/floor bottom; "
                "set the connected bottom length to thigh/knee or omit socks"
            )
        hard_normal = list(TEMPLATE["hard_boxes"]["socks"])
        sock_length = self._choice(
            layout,
            "length",
            "mid_calf",
            {"ankle", "crew", "mid_calf", "knee", "knee_high"},
        )
        start_y = {
            "knee": 0.735,
            "knee_high": 0.735,
            "mid_calf": 0.765,
            "crew": 0.80,
            "ankle": 0.835,
        }.get(sock_length, 0.765)
        if bottom_length in TEMPLATE["landmarks"]["bottom_hem_y"]:
            start_y = max(start_y, TEMPLATE["landmarks"]["bottom_hem_y"][bottom_length] + 0.018)
        hard_normal[1] = min(start_y, hard_normal[3] - 0.035)
        hard = self._normal_box_to_pixels(tuple(hard_normal), width, height)
        box_width, box_height = hard[2] - hard[0], hard[3] - hard[1]
        scale = min(box_width / cutout.width, box_height / cutout.height)
        target_width = max(1, int(round(cutout.width * scale)))
        target_height = max(1, int(round(cutout.height * scale)))
        x = hard[0] + (box_width - target_width) // 2
        y = hard[3] - target_height
        return (x, y, target_width, target_height), hard, f"socks {sock_length}"

    def _box_placement(self, slot, cutout, layout, width, height):
        hard = self._normal_box_to_pixels(TEMPLATE["hard_boxes"][slot], width, height)
        box_width, box_height = hard[2] - hard[0], hard[3] - hard[1]
        default_size = "large" if slot == "shoes" else "medium"
        size = self._choice(layout, "size", default_size, {"tiny", "small", "medium", "large"})
        size_multiplier = {"tiny": 0.58, "small": 0.72, "medium": 0.86, "large": 1.0}.get(size, 0.86)
        scale = min(box_width * size_multiplier / cutout.width, box_height * size_multiplier / cutout.height)
        target_width = max(1, int(round(cutout.width * scale)))
        target_height = max(1, int(round(cutout.height * scale)))
        x = hard[0] + (box_width - target_width) // 2
        y = hard[1] + (box_height - target_height) // 2
        if slot in {"hat", "necklace"}:
            y = hard[1]
        elif slot in {"bag", "shoes"}:
            y = hard[3] - target_height
        return (x, y, target_width, target_height), hard, f"{slot} {size}"

    @staticmethod
    def _split_pair(cutout):
        """Split a side-by-side accessory pair at the quietest centre column."""
        alpha = np.asarray(cutout.getchannel("A"))
        projection = np.count_nonzero(alpha >= 8, axis=0)
        if cutout.width < 4:
            return [cutout]
        start = max(1, int(cutout.width * 0.30))
        end = min(cutout.width - 1, int(cutout.width * 0.70))
        split = start + int(np.argmin(projection[start:end])) if end > start else cutout.width // 2
        # A single/connected accessory has foreground through the centre. Do
        # not cut it merely because one central column happens to be thinner.
        quiet_limit = max(1, int(round(cutout.height * 0.02)))
        if projection[split] > quiet_limit:
            return [cutout]
        parts = []
        for x0, x1 in ((0, split), (split, cutout.width)):
            part = cutout.crop((x0, 0, x1, cutout.height))
            bbox = part.getchannel("A").point(lambda value: 255 if value >= 8 else 0).getbbox()
            if bbox:
                parts.append(part.crop(bbox))
        if len(parts) != 2:
            return [cutout]
        total_foreground = max(1, int(np.count_nonzero(alpha >= 8)))
        part_foreground = [
            int(np.count_nonzero(np.asarray(part.getchannel("A")) >= 8)) for part in parts
        ]
        if min(part_foreground) < total_foreground * 0.15:
            return [cutout]
        return parts

    def _earrings_placements(self, cutout, layout, width, height):
        parts = self._split_pair(cutout)
        if len(parts) == 1:
            side = self._choice(layout, "side", "right", {"left", "right"})
            box_names = ("earrings_left",) if side == "left" else ("earrings_right",)
        else:
            box_names = ("earrings_left", "earrings_right")
        size = self._choice(layout, "size", "medium", {"tiny", "small", "medium", "large"})
        multiplier = {"tiny": 0.58, "small": 0.72, "medium": 0.86, "large": 1.0}.get(size, 0.86)
        results = []
        for index, part in enumerate(parts[:2]):
            box_name = box_names[index]
            hard = self._normal_box_to_pixels(TEMPLATE["hard_boxes"][box_name], width, height)
            box_width, box_height = hard[2] - hard[0], hard[3] - hard[1]
            scale = min(box_width * multiplier / part.width, box_height * multiplier / part.height)
            target_width = max(1, int(round(part.width * scale)))
            target_height = max(1, int(round(part.height * scale)))
            x = hard[0] + (box_width - target_width) // 2
            y = hard[1] + (box_height - target_height) // 2
            side_label = "L" if box_name.endswith("left") else "R"
            results.append(
                (part, (x, y, target_width, target_height), hard, f"earrings {side_label}/{size}")
            )
        return results

    @staticmethod
    def _composite(canvas, cutout, placement):
        x, y, target_width, target_height = placement
        resized = cutout.resize((target_width, target_height), Image.Resampling.LANCZOS)
        shadow = Image.new("RGBA", resized.size, (0, 0, 0, 0))
        shadow_alpha = resized.getchannel("A").filter(ImageFilter.GaussianBlur(2)).point(lambda value: value // 9)
        shadow.putalpha(shadow_alpha)
        canvas.alpha_composite(shadow, (x + 2, y + 3))
        canvas.alpha_composite(resized, (x, y))
        return (x, y, x + target_width, y + target_height)

    @staticmethod
    def _dashed_line(draw, xy, fill, width=1, dash=7, gap=5):
        x0, y0, x1, y1 = xy
        if y0 == y1:
            x = x0
            while x < x1:
                draw.line((x, y0, min(x + dash, x1), y1), fill=fill, width=width)
                x += dash + gap
        else:
            y = y0
            while y < y1:
                draw.line((x0, y, x1, min(y + dash, y1)), fill=fill, width=width)
                y += dash + gap

    def _draw_guides(self, canvas, records, width, height):
        draw = ImageDraw.Draw(canvas)
        line_width = max(1, width // 384)
        red = (255, 54, 54, 255)
        cyan = (0, 155, 210, 255)
        gray = (125, 125, 125, 210)
        landmarks = TEMPLATE["landmarks"]

        horizontal = {
            "neck": landmarks["neck_y"],
            "shoulder": landmarks["shoulder_y"],
            "high waist": landmarks["bottom_waist_y"]["high"],
            "knee": landmarks["bottom_hem_y"]["knee"],
            "ankle": landmarks["bottom_hem_y"]["ankle"],
        }
        x0 = int(width * 0.12)
        x1 = int(width * 0.72)
        for name, normal_y in horizontal.items():
            y = int(round(normal_y * height))
            self._dashed_line(draw, (x0, y, x1, y), gray, line_width)
            draw.text((x0 + 3, y + 2), name, fill=gray)

        centre_x = int(round(TEMPLATE["body_center_x"] * width))
        self._dashed_line(
            draw,
            (centre_x, int(height * 0.14), centre_x, int(height * 0.985)),
            gray,
            line_width,
        )
        for hard, actual, label in records:
            draw.rectangle(hard, outline=red, width=line_width)
            draw.rectangle(actual, outline=cyan, width=line_width)
            draw.text((hard[0] + 4, hard[1] + 4), label, fill=red)

    def compose(self, outfit_spec, width, height, background, show_layout_guides=False, **images):
        items = self._parse(outfit_spec)
        colors = {
            "white": (255, 255, 255),
            "off_white": (248, 248, 245),
            "light_gray": (238, 238, 238),
        }
        canvas = Image.new("RGBA", (width, height), colors[background] + (255,))
        records = []
        layouts = {slot: self._layout(items.get(slot, {})) for slot in SLOTS}
        for slot, layout in layouts.items():
            declared_slot = layout.get("slot")
            if declared_slot is not None and declared_slot != slot:
                raise ValueError(
                    f"{layout.get('item_id', slot)} is under items.{slot} but declares slot={declared_slot!r}"
                )

        for slot in DRAW_ORDER:
            tensor = images.get(slot)
            if tensor is None:
                continue
            layout = layouts[slot]
            cutout = self._cutout(self._image(tensor), slot, layout)
            if cutout is None:
                raise ValueError(
                    f"{slot} image has no detectable foreground; check the image/background "
                    "or set background_threshold in that item's JSON"
                )

            if slot == "top":
                placement, hard, label = self._top_placement(cutout, layout, width, height)
            elif slot == "bottom":
                placement, hard, label = self._bottom_placement(cutout, layout, width, height)
            elif slot == "socks":
                placement, hard, label = self._socks_placement(
                    cutout,
                    layout,
                    layouts.get("bottom", {}),
                    width,
                    height,
                    images.get("bottom") is not None,
                )
            elif slot == "earrings":
                for part, placement, hard, label in self._earrings_placements(
                    cutout, layout, width, height
                ):
                    actual = self._composite(canvas, part, placement)
                    records.append((hard, actual, label))
                continue
            else:
                placement, hard, label = self._box_placement(slot, cutout, layout, width, height)
            actual = self._composite(canvas, cutout, placement)
            records.append((hard, actual, label))

        if show_layout_guides:
            self._draw_guides(canvas, records, width, height)
        output = torch.from_numpy(np.asarray(canvas.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
        return (output,)


NODE_CLASS_MAPPINGS = {"OutfitReferenceComposer": OutfitReferenceComposer}
NODE_DISPLAY_NAME_MAPPINGS = {"OutfitReferenceComposer": "Outfit Reference Composer"}
