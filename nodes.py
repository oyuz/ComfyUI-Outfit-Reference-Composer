import json
from collections import deque

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter


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

# V2 deliberately uses the input socket name as its only semantic signal.
# Every box is permanent: an empty slot stays empty and never causes another
# product to move.  Neck accessories deliberately overlap the upper top box
# and are rendered later; every other pair remains disjoint. Coordinates are
# normalized so the same template works at every supported resolution.
V2_OUTFIT_CENTER_X = 0.46
V2_FIXED_SLOTS = {
    "hat": (0.31, 0.005, 0.61, 0.101),
    "glasses": (0.32, 0.102, 0.60, 0.168),
    "earrings_left": (0.23, 0.103, 0.32, 0.158),
    "earrings_right": (0.60, 0.103, 0.69, 0.158),
    "necklace": (0.35, 0.170, 0.57, 0.260),
    "top": (0.16, 0.175, 0.76, 0.475),
    "bottom": (0.20, 0.480, 0.72, 0.815),
    "bracelet": (0.01, 0.525, 0.17, 0.645),
    "bag": (0.73, 0.505, 0.99, 0.755),
    # Socks share the lower band beside footwear instead of reserving a full
    # row. This keeps an empty socks input from creating a large vertical gap
    # and gives tall boots substantially more height without any overlap.
    "socks": (0.09, 0.820, 0.25, 0.995),
    "shoes": (0.265, 0.820, 0.655, 0.995),
}

V2_DRAW_ORDER = (
    "top",
    "bottom",
    "socks",
    "shoes",
    "hat",
    "earrings",
    "glasses",
    "necklace",
    "bag",
    "bracelet",
)

# V3 starts from V2's exact tight-cutout placements, then resolves the two
# semantically ambiguous garment boundaries against the *actual neighbouring
# item*.  These fractions are object-relative, not canvas hard-box
# coordinates: a top_coverwaist top reaches 30% into the placed bottom, and a
# bottom_shoehalf bottom reaches the placed footwear's two-thirds line plus a tiny
# seam guard.  Only the lower third of the shoe layer is retained so wide
# shoes cannot leak around narrow trouser legs above the intended toe band.
V3_TOP_COVERWAIST_HIP_FRACTION = 0.30
V3_BOTTOM_SHOEHALF_COVER_FRACTION = 2.0 / 3.0
# The footwear crop still starts at the exact two-thirds boundary.  Extend the
# visible trouser hem a few pixels past that line so antialiasing and curved
# shoe/toe silhouettes cannot expose the canvas between the two layers.
V3_BOTTOM_SHOEHALF_SEAM_OVERLAP_FRACTION = 0.03
V3_LAYOUT_TAGS = {
    "top": "top_coverwaist",
    "bottom": "bottom_shoehalf",
}

# V3_TOP is a separate node, not a mode switch on V2. It keeps the same
# cutout/contain rules but dedicates the full portrait canvas to upper-body
# products only. No lower-body or side-accessory sockets are exposed.
V3_TOP_SLOTS = (
    "top",
    "hat",
    "glasses",
    "earrings",
    "necklace",
)

V3_TOP_CENTER_X = 0.50
V3_TOP_FIXED_SLOTS = {
    # Preserve the visually successful V2 upper-body proportions while
    # expanding that region to the full V3_TOP canvas. Headwear and neckwear
    # therefore grow in step with the much larger top instead of appearing
    # undersized beside it.
    "hat": (0.18, 0.005, 0.82, 0.225),
    "glasses": (0.25, 0.230, 0.75, 0.335),
    "earrings_left": (0.07, 0.230, 0.25, 0.335),
    "earrings_right": (0.75, 0.230, 0.93, 0.335),
    # Neckwear keeps Round 13's lower centre point, but is reined in one step
    # so chains and scarves remain subordinate to the garment neckline.
    "necklace": (0.28, 0.350, 0.72, 0.550),
    "top": (0.015, 0.345, 0.985, 0.995),
}

V3_TOP_DRAW_ORDER = (
    "top",
    "hat",
    "earrings",
    "glasses",
    "necklace",
)

# V3/V3_TOP keep the proven full-set coordinates above, but collapse empty
# semantic rows before fitting individual cutouts.  The row allocator only
# changes the vertical limits: source aspect ratios, centre axes and draw order
# remain unchanged.  This makes a missing face row useful to the hat, for
# example, without moving unrelated side accessories into arbitrary positions.
V3_ROW_GAP = 0.005
V3_FULL_HEAD_ROWS = {
    "hat": (0.005, 0.101),
    "face": (0.102, 0.168),
}
V3_FULL_BODY_ROWS = {
    "top": (0.175, 0.475),
    "bottom": (0.480, 0.815),
    "foot": (0.820, 0.995),
}
V3_TOP_ROWS = {
    "hat": (0.005, 0.225),
    "face": (0.230, 0.335),
    "top": (0.345, 0.995),
}

# Content cropping is deliberately conservative.  Padding is measured against
# the requested canvas and the final extent is rounded *outwards* to a multiple
# that is friendly to common diffusion/VAE pipelines.
V3_CROP_MULTIPLE = 64
V3_CROP_MIN_PADDING = 16
V3_CROP_PADDING_FRACTION = 0.015


def _allocate_active_rows(rows, active_names, gap=V3_ROW_GAP):
    """Collapse missing semantic rows while preserving the all-row template."""
    active = [name for name in rows if name in active_names]
    if not active:
        return {}
    if len(active) == len(rows):
        return {name: tuple(rows[name]) for name in active}

    span_start = min(bounds[0] for bounds in rows.values())
    span_end = max(bounds[1] for bounds in rows.values())
    available = max(0.0, span_end - span_start - gap * (len(active) - 1))
    original_heights = {
        name: max(1e-9, rows[name][1] - rows[name][0]) for name in active
    }
    total_weight = sum(original_heights.values())
    allocated = {}
    cursor = span_start
    for index, name in enumerate(active):
        if index == len(active) - 1:
            end = span_end
        else:
            end = cursor + available * original_heights[name] / total_weight
        allocated[name] = (cursor, end)
        cursor = end + gap
    return allocated


def _remap_box_y(box, source_row, target_row):
    """Map a box's vertical offsets from one semantic row into another."""
    x0, y0, x1, y1 = box
    source_start, source_end = source_row
    target_start, target_end = target_row
    source_height = max(1e-9, source_end - source_start)
    scale = (target_end - target_start) / source_height
    return (
        x0,
        target_start + (y0 - source_start) * scale,
        x1,
        target_start + (y1 - source_start) * scale,
    )


def _fit_cutout_in_box(cutout, hard, slot_padding, horizontal_anchor=None):
    """Contain a tight cutout uniformly inside an explicit pixel box."""
    box_width = max(1, hard[2] - hard[0])
    box_height = max(1, hard[3] - hard[1])
    requested_padding = max(0, int(slot_padding))
    padding = min(
        requested_padding,
        max(0, (min(box_width, box_height) - 1) // 2),
    )
    inner_x0 = hard[0] + padding
    inner_y0 = hard[1] + padding
    inner_width = max(1, box_width - padding * 2)
    inner_height = max(1, box_height - padding * 2)
    scale = min(
        inner_width / max(1, cutout.width),
        inner_height / max(1, cutout.height),
    )
    axis_x = (hard[0] + hard[2]) / 2
    if horizontal_anchor is not None:
        scale = min(
            scale,
            (axis_x - inner_x0) / max(1, horizontal_anchor),
            (inner_x0 + inner_width - axis_x)
            / max(1, cutout.width - horizontal_anchor),
        )
    target_width = min(
        max(1, int(round(cutout.width * scale))),
        inner_width,
    )
    target_height = min(
        max(1, int(round(cutout.height * scale))),
        inner_height,
    )
    if horizontal_anchor is None:
        x = inner_x0 + (inner_width - target_width) // 2
    else:
        x = int(round(axis_x - horizontal_anchor * target_width / cutout.width))
        x = min(max(x, inner_x0), inner_x0 + inner_width - target_width)
    y = inner_y0 + (inner_height - target_height) // 2
    return (x, y, target_width, target_height)


def _add_cutout_to_occupancy(occupancy, cutout, placement):
    """Union a placed cutout's real alpha into a canvas-sized occupancy mask."""
    x, y, target_width, target_height = placement
    resized_alpha = cutout.getchannel("A").resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )
    left = max(0, x)
    top = max(0, y)
    right = min(occupancy.width, x + target_width)
    bottom = min(occupancy.height, y + target_height)
    if left >= right or top >= bottom:
        return
    source = resized_alpha.crop((left - x, top - y, right - x, bottom - y))
    current = occupancy.crop((left, top, right, bottom))
    occupancy.paste(ImageChops.lighter(current, source), (left, top))


def _content_crop_box(occupancy, slot_padding=0, multiple=V3_CROP_MULTIPLE):
    """Return a padded, outward-aligned crop around alpha-visible products."""
    visible = occupancy.point(lambda value: 255 if value >= 8 else 0).getbbox()
    if not visible:
        return (0, 0, occupancy.width, occupancy.height)
    padding = max(
        V3_CROP_MIN_PADDING,
        int(slot_padding) * 2,
        int(round(min(occupancy.size) * V3_CROP_PADDING_FRACTION)),
    )
    wanted = (
        max(0, visible[0] - padding),
        max(0, visible[1] - padding),
        min(occupancy.width, visible[2] + padding),
        min(occupancy.height, visible[3] + padding),
    )

    def aligned_extent(start, end, limit):
        wanted_size = max(1, end - start)
        if wanted_size >= limit:
            return 0, limit
        aligned_size = min(
            limit,
            int(np.ceil(wanted_size / max(1, multiple))) * max(1, multiple),
        )
        centre = (start + end) / 2
        aligned_start = int(round(centre - aligned_size / 2))
        aligned_start = min(max(0, aligned_start), limit - aligned_size)
        return aligned_start, aligned_start + aligned_size

    x0, x1 = aligned_extent(wanted[0], wanted[2], occupancy.width)
    y0, y1 = aligned_extent(wanted[1], wanted[3], occupancy.height)
    return (x0, y0, x1, y1)


def _canvas_to_rgb_tensor(canvas, occupancy=None, crop=False, slot_padding=0):
    if crop and occupancy is not None:
        canvas = canvas.crop(_content_crop_box(occupancy, slot_padding))
    return torch.from_numpy(
        np.asarray(canvas.convert("RGB")).astype(np.float32) / 255.0
    ).unsqueeze(0)

# All coordinates are normalized to the output canvas. The main outfit is
# slightly left of centre so the bag and wrist accessory have a dedicated
# gutter on the right. Main garments are measured first and then packed into
# the main column without changing their source aspect ratio.
TEMPLATE = {
    "body_center_x": 0.45,
    "main_top_y": 0.18,
    "main_bottom_y": 0.99,
    "hard_boxes": {
        # For the packed main column only the horizontal limits are static;
        # guide rectangles get their vertical ranges after packing.
        "top": (0.10, 0.18, 0.74, 0.99),
        "bottom": (0.18, 0.18, 0.72, 0.99),
        "socks": (0.27, 0.18, 0.63, 0.99),
        # The target character is deliberately stylised (roughly five heads
        # tall), so headwear and footwear need more visual weight than they
        # would in a realistic adult template.
        "shoes": (0.22, 0.855, 0.68, 0.99),
        "hat": (0.29, 0.005, 0.61, 0.10),
        "glasses": (0.30, 0.105, 0.60, 0.17),
        # A paired earring image is split and placed on the two sides of the
        # glasses. The three boxes are disjoint.
        "earrings_left": (0.20, 0.105, 0.29, 0.17),
        "earrings_right": (0.61, 0.105, 0.70, 0.17),
        "necklace": (0.33, 0.18, 0.57, 0.31),
        "bracelet_left": (0.025, 0.50, 0.225, 0.61),
        "bracelet_right": (0.775, 0.50, 0.975, 0.61),
        "bag": (0.755, 0.50, 0.985, 0.75),
    },
    "landmarks": {
        # shoulder_y and hip_y are diagnostic body axes. Garments retain
        # local shoulder/hip anchors, then the no-overlap collage is packed.
        "neck_y": 0.18,
        "shoulder_y": 0.23,
        "hip_y": 0.49,
        "bag_handle_y": 0.52,
        "wrist_y": 0.555,
        # The top edge of a tight bottom cutout is treated as its waistband.
        # Each waist level is expressed as a distance above the shared hip
        # axis so high/mid/low remain meaningful even when no top is present.
        "waist_offset_from_hip": {
            "high": 0.11,
            "mid": 0.09,
            "low": 0.07,
        },
        "top_height": {
            "crop": 0.22,
            "waist": 0.275,
            "hip": 0.325,
            "upper_thigh": 0.37,
        },
        "bottom_height": {
            "thigh": 0.17,
            "knee": 0.23,
            "calf": 0.30,
            "ankle": 0.36,
            "floor": 0.39,
        },
        # These are shoulder spans, not full sleeve-to-sleeve widths.
        "shoulder_width": {
            "slim": 0.20,
            "regular": 0.225,
            "loose": 0.255,
            "oversized": 0.30,
        },
        "hip_width": {
            "slim": 0.225,
            "regular": 0.255,
            "loose": 0.29,
            "oversized": 0.325,
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
    def _uniform_size(cutout, scale, max_width, max_height):
        """Return a bounded size produced by one scale on both axes."""
        scale = min(
            max(float(scale), 1.0 / max(cutout.width, cutout.height)),
            float(max_width) / max(1, cutout.width),
            float(max_height) / max(1, cutout.height),
        )
        return (
            max(1, int(round(cutout.width * scale))),
            max(1, int(round(cutout.height * scale))),
        )

    @staticmethod
    def _weighted_scale(width_scale, height_scale, width_weight=0.65):
        """Blend two semantic scale targets while keeping a single scale."""
        width_scale = max(float(width_scale), 1e-6)
        height_scale = max(float(height_scale), 1e-6)
        height_weight = 1.0 - width_weight
        return float(
            np.exp(width_weight * np.log(width_scale) + height_weight * np.log(height_scale))
        )

    @staticmethod
    def _band_span(cutout, start_ratio, end_ratio, percentile=75):
        """Measure a stable foreground width inside a vertical alpha band."""
        alpha = np.asarray(cutout.getchannel("A")) >= 8
        start = max(0, min(cutout.height - 1, int(round(cutout.height * start_ratio))))
        end = max(start + 1, min(cutout.height, int(round(cutout.height * end_ratio))))
        spans = []
        for row in alpha[start:end]:
            columns = np.flatnonzero(row)
            if columns.size:
                spans.append(int(columns[-1] - columns[0] + 1))
        if not spans:
            return max(1, cutout.width)
        return max(1, int(round(np.percentile(spans, percentile))))

    def _top_placement(self, cutout, layout, width, height):
        landmarks = TEMPLATE["landmarks"]
        fit = self._choice(
            layout,
            "fit",
            "regular",
            set(landmarks["shoulder_width"]),
            {"oversize": "oversized"},
        )
        length = self._choice(layout, "length", "waist", set(landmarks["top_height"]))
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
        hard = self._normal_box_to_pixels(TEMPLATE["hard_boxes"]["top"], width, height)
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
        source_shoulder = max(1.0, cutout.width * shoulder_ratio)
        width_scale = target_shoulder / source_shoulder
        desired_height = landmarks["top_height"].get(
            length, landmarks["top_height"]["waist"]
        ) * height
        height_scale = desired_height / max(1.0, cutout.height)
        semantic_scale = self._weighted_scale(width_scale, height_scale, 0.65)
        target_width, target_height = self._uniform_size(
            cutout,
            semantic_scale,
            hard_width,
            min(hard_height, int(round(height * 0.40))),
        )
        centre_x = TEMPLATE["body_center_x"] * width
        x = int(round(centre_x - target_width / 2))
        y = int(round(TEMPLATE["main_top_y"] * height))
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
        garment_type = self._choice(
            layout, "garment_type", "", {"", "shorts", "pants", "skirt"}
        )
        waist = self._choice(layout, "waist", "mid", {"high", "mid", "low"})
        length = self._choice(
            layout,
            "length",
            "floor",
            set(landmarks["bottom_height"]),
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
        hard = self._normal_box_to_pixels(TEMPLATE["hard_boxes"]["bottom"], width, height)
        hard_width, hard_height = hard[2] - hard[0], hard[3] - hard[1]
        silhouette_multiplier = {
            "slim": 0.94,
            "straight": 1.00,
            "wide": 1.04,
            "baggy": 1.08,
            "flare": 1.04,
            "skirt": 1.00,
            "pleated": 1.03,
            "a_line": 1.06,
            "voluminous": 1.08,
        }.get(silhouette, 1.0)

        if "source_hip_width_ratio" in layout:
            try:
                hip_ratio = float(layout["source_hip_width_ratio"])
            except (TypeError, ValueError) as error:
                raise ValueError("source_hip_width_ratio must be a number from 0.10 to 1.0") from error
            if not 0.10 <= hip_ratio <= 1.0:
                raise ValueError("source_hip_width_ratio must be a number from 0.10 to 1.0")
            source_hip_span = cutout.width * hip_ratio
        else:
            band = (0.12, 0.34) if garment_type == "skirt" else (0.15, 0.36)
            source_hip_span = self._band_span(cutout, *band)

        target_hip = (
            landmarks["hip_width"].get(fit, landmarks["hip_width"]["regular"])
            * silhouette_multiplier
            * width
        )
        width_scale = target_hip / max(1.0, source_hip_span)
        desired_height = landmarks["bottom_height"].get(
            length, landmarks["bottom_height"]["floor"]
        ) * height
        height_scale = desired_height / max(1.0, cutout.height)
        semantic_scale = self._weighted_scale(width_scale, height_scale, 0.65)
        target_width, target_height = self._uniform_size(
            cutout,
            semantic_scale,
            hard_width,
            min(hard_height, int(round(height * 0.42))),
        )
        centre_x = TEMPLATE["body_center_x"] * width
        x = int(round(centre_x - target_width / 2))
        hip_y = landmarks["hip_y"] * height
        waist_offset = landmarks["waist_offset_from_hip"][waist] * height
        y = int(round(hip_y - waist_offset))
        x = min(max(x, hard[0]), hard[2] - target_width)
        y = min(max(y, hard[1]), hard[3] - target_height)
        label = f"bottom {waist}/{length}/{silhouette}"
        return (x, y, target_width, target_height), hard, label

    def _socks_placement(self, cutout, layout, bottom_layout, width, height, bottom_present=False):
        bottom_length = self._choice(
            bottom_layout,
            "length",
            "" if not bottom_present else "floor",
            {"", *TEMPLATE["landmarks"]["bottom_height"]},
            {"full": "floor"},
        )
        if bottom_present and not bottom_length:
            bottom_length = "floor"
        if bottom_present and bottom_length not in {"thigh", "knee"}:
            raise ValueError(
                "socks cannot share the leg region with a calf/ankle/floor bottom; "
                "set the connected bottom length to thigh/knee or omit socks"
            )
        sock_length = self._choice(
            layout,
            "length",
            "mid_calf",
            {"ankle", "crew", "mid_calf", "knee", "knee_high"},
        )
        hard = self._normal_box_to_pixels(TEMPLATE["hard_boxes"]["socks"], width, height)
        box_width, box_height = hard[2] - hard[0], hard[3] - hard[1]
        desired_height = {
            "knee": 0.145,
            "knee_high": 0.145,
            "mid_calf": 0.12,
            "crew": 0.095,
            "ankle": 0.07,
        }.get(sock_length, 0.12) * height
        target_width, target_height = self._uniform_size(
            cutout,
            desired_height / max(1.0, cutout.height),
            box_width,
            min(box_height, int(round(height * 0.15))),
        )
        x = hard[0] + (box_width - target_width) // 2
        y = int(round(0.72 * height))
        y = min(max(y, hard[1]), hard[3] - target_height)
        return (x, y, target_width, target_height), hard, f"socks {sock_length}"

    def _box_placement(self, slot, cutout, layout, width, height):
        box_name = slot
        side = None
        if slot == "bracelet":
            raw_side = str(layout.get("side", layout.get("placement", "left"))).strip().lower()
            side_aliases = {
                "wrist_side": "left",
                "left_wrist": "left",
                "left_wrist_side": "left",
                "right_wrist": "right",
                "right_wrist_side": "right",
            }
            side = side_aliases.get(raw_side, raw_side)
            if side not in {"left", "right"}:
                item_id = layout.get("item_id", "bracelet")
                raise ValueError(
                    f"{item_id}.side/placement must be left or right wrist; got {raw_side!r}"
                )
            box_name = f"bracelet_{side}"

        hard = self._normal_box_to_pixels(TEMPLATE["hard_boxes"][box_name], width, height)
        box_width, box_height = hard[2] - hard[0], hard[3] - hard[1]
        default_size = "large" if slot == "shoes" else "medium"
        size = self._choice(layout, "size", default_size, {"tiny", "small", "medium", "large"})
        if slot == "glasses":
            size_multiplier = {
                "tiny": 0.76,
                "small": 0.90,
                "medium": 0.96,
                "large": 1.0,
            }.get(size, 0.96)
        elif slot == "hat":
            size_multiplier = {
                "tiny": 0.76,
                "small": 0.86,
                "medium": 0.95,
                "large": 1.0,
            }.get(size, 0.95)
        elif slot == "bracelet":
            # Bracelet and bead-strand product shots often occupy very
            # different fractions of their source canvas. Normalise them by
            # visible ring height, with only a restrained semantic size range.
            desired_height = {
                "tiny": 0.043,
                "small": 0.047,
                "medium": 0.052,
                "large": 0.057,
            }.get(size, 0.052) * height
            target_width, target_height = self._uniform_size(
                cutout,
                desired_height / max(1.0, cutout.height),
                box_width,
                box_height,
            )
            x = hard[0] + (box_width - target_width) // 2
            wrist_y = int(round(TEMPLATE["landmarks"]["wrist_y"] * height))
            y = wrist_y - target_height // 2
            y = min(max(y, hard[1]), hard[3] - target_height)
            return (
                (x, y, target_width, target_height),
                hard,
                f"bracelet {side}/{size}",
            )
        else:
            size_multiplier = {
                "tiny": 0.58,
                "small": 0.72,
                "medium": 0.86,
                "large": 1.0,
            }.get(size, 0.86)
        scale = min(box_width * size_multiplier / cutout.width, box_height * size_multiplier / cutout.height)
        target_width = max(1, int(round(cutout.width * scale)))
        target_height = max(1, int(round(cutout.height * scale)))
        x = hard[0] + (box_width - target_width) // 2
        y = hard[1] + (box_height - target_height) // 2
        if slot in {"hat", "necklace"}:
            y = hard[1]
        elif slot == "bag":
            # Product shots include the handle. Its top edge is the useful
            # hand-held anchor; bottom-aligning made small bags float near the
            # knee rather than hang from the hand.
            y = int(round(TEMPLATE["landmarks"]["bag_handle_y"] * height))
            y = min(max(y, hard[1]), hard[3] - target_height)
        elif slot == "shoes":
            y = hard[3] - target_height
        return (x, y, target_width, target_height), hard, f"{slot} {size}"

    def _main_gap(self, previous, current, layouts, height):
        """Return a semantic but compact gap between two main-column slots."""
        if previous == "top" and current == "bottom":
            top = layouts.get("top", {})
            bottom = layouts.get("bottom", {})
            coverage = str(top.get("coverage", "")).strip().lower()
            top_length = str(top.get("length", "waist")).strip().lower()
            if coverage == "show_midriff" or top_length == "crop":
                waist = str(bottom.get("waist", "mid")).strip().lower()
                fraction = {"high": 0.028, "mid": 0.036, "low": 0.045}.get(waist, 0.036)
            else:
                fraction = 0.012
        elif previous == "bottom" and current == "socks":
            fraction = 0.010
        elif previous == "socks" and current == "shoes":
            fraction = 0.008
        elif previous == "bottom" and current == "shoes":
            length = str(layouts.get("bottom", {}).get("length", "floor")).strip().lower()
            fraction = {
                "thigh": 0.070,
                "knee": 0.055,
                "calf": 0.032,
                "ankle": 0.012,
                "floor": 0.010,
                "full": 0.010,
            }.get(length, 0.020)
        else:
            fraction = 0.012
        return max(8, int(round(height * fraction)))

    def _pack_main(self, placements, layouts, width, height):
        """Resolve semantic main-slot anchors without distorting any item.

        A connected clothing chain is kept compact, while a missing middle
        category creates a real body-region break. For example, top + shoes
        leaves the shoes at the foot anchor instead of pulling them below the
        top. If a connected chain is too tall, every present main item receives
        the same additional scale.
        """
        sequence = [slot for slot in ("top", "bottom", "socks", "shoes") if slot in placements]
        if not sequence:
            return placements
        main_bottom = int(round(TEMPLATE["main_bottom_y"] * height))
        def place_at(global_scale):
            packed = dict(placements)
            gaps_before = {}
            for slot in sequence:
                placement, hard, label = placements[slot]
                target_width = max(1, int(round(placement[2] * global_scale)))
                target_height = max(1, int(round(placement[3] * global_scale)))
                x0, _, x1, _ = hard
                centre_x = TEMPLATE["body_center_x"] * width
                x = int(round(centre_x - target_width / 2))
                x = min(max(x, x0), x1 - target_width)
                if (
                    slot == "shoes"
                    and "bottom" not in placements
                    and "socks" not in placements
                    and global_scale == 1.0
                ):
                    # Preserve the exact fixed foot anchor (including integer
                    # rounding) when there is no connected lower-body chain.
                    x = placement[0]

                desired_y = placement[1]
                previous = None
                if slot == "bottom" and "top" in packed and "top" in placements:
                    previous = "top"
                elif slot == "socks" and "bottom" in packed and "bottom" in placements:
                    previous = "bottom"
                elif slot == "shoes":
                    if "socks" in packed and "socks" in placements:
                        previous = "socks"
                    elif "bottom" in packed and "bottom" in placements:
                        previous = "bottom"
                    else:
                        # Isolated shoes stay bottom-aligned in the fixed foot
                        # region even if some unrelated main item was scaled.
                        desired_y = hard[3] - target_height

                if previous is None:
                    y = desired_y
                    gap_before = 0
                else:
                    previous_box = packed[previous][0]
                    gap_before = self._main_gap(previous, slot, layouts, height)
                    compact_y = previous_box[1] + previous_box[3] + gap_before
                    # The waistband is a real hip-relative anchor. Socks and
                    # shoes use their predecessor as the stronger anchor once
                    # the corresponding garment chain exists.
                    y = max(desired_y, compact_y) if slot == "bottom" else compact_y

                actual = (x, int(round(y)), target_width, target_height)
                gaps_before[slot] = gap_before
                packed[slot] = (actual, hard, label)

            max_end = max(packed[slot][0][1] + packed[slot][0][3] for slot in sequence)
            return packed, gaps_before, max_end

        global_scale = 1.0
        packed, gaps_before, max_end = place_at(global_scale)
        if max_end > main_bottom:
            low, high = 0.0, 1.0
            for _ in range(28):
                middle = (low + high) / 2
                candidate, candidate_gaps, candidate_end = place_at(middle)
                if candidate_end <= main_bottom:
                    low = middle
                    packed, gaps_before, max_end = candidate, candidate_gaps, candidate_end
                else:
                    high = middle

        main_top = int(round(TEMPLATE["main_top_y"] * height))
        for slot in sequence:
            actual, hard, label = packed[slot]
            x0, _, x1, _ = hard
            pad = min(3, max(0, gaps_before[slot] // 2))
            dynamic_hard = (
                x0,
                max(main_top, actual[1] - pad),
                x1,
                min(main_bottom, actual[1] + actual[3] + 3),
            )
            packed[slot] = (actual, dynamic_hard, label)
        return packed

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
        if min(cutout.width, cutout.height, target_width, target_height) >= 8:
            source_aspect = cutout.width / cutout.height
            target_aspect = target_width / target_height
            axis_change = max(source_aspect / target_aspect, target_aspect / source_aspect)
            if axis_change > 1.04:
                raise ValueError(
                    f"non-uniform resize is forbidden (axis ratio changed {axis_change:.3f}x)"
                )
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
            "hip axis": landmarks["hip_y"],
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

        cutouts = {}
        for slot in SLOTS:
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
            cutouts[slot] = cutout

        placements = {}
        for slot, cutout in cutouts.items():
            layout = layouts[slot]
            if slot == "top":
                placements[slot] = self._top_placement(cutout, layout, width, height)
            elif slot == "bottom":
                placements[slot] = self._bottom_placement(cutout, layout, width, height)
            elif slot == "socks":
                placements[slot] = self._socks_placement(
                    cutout,
                    layout,
                    layouts.get("bottom", {}),
                    width,
                    height,
                    images.get("bottom") is not None,
                )
            elif slot == "earrings":
                continue
            else:
                placements[slot] = self._box_placement(slot, cutout, layout, width, height)

        placements = self._pack_main(placements, layouts, width, height)

        for slot in DRAW_ORDER:
            cutout = cutouts.get(slot)
            if cutout is None:
                continue
            layout = layouts[slot]
            if slot == "earrings":
                for part, placement, hard, label in self._earrings_placements(
                    cutout, layout, width, height
                ):
                    actual = self._composite(canvas, part, placement)
                    records.append((hard, actual, label))
                continue
            placement, hard, label = placements[slot]
            actual = self._composite(canvas, cutout, placement)
            records.append((hard, actual, label))

        if show_layout_guides:
            self._draw_guides(canvas, records, width, height)
        output = torch.from_numpy(np.asarray(canvas.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
        return (output,)


class OutfitReferenceComposerV2(OutfitReferenceComposer):
    """Maximise tight product cutouts inside permanent, metadata-free slots."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "width": ("INT", {"default": 1536, "min": 256, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 2048, "min": 256, "max": 2048, "step": 64}),
            "background": (["off_white", "white", "light_gray"], {"default": "off_white"}),
            "background_threshold": (
                "INT",
                {"default": 24, "min": 4, "max": 100, "step": 1},
            ),
            "slot_padding": (
                "INT",
                {"default": 10, "min": 0, "max": 64, "step": 1},
            ),
            "show_layout_guides": ("BOOLEAN", {"default": False}),
        }
        return {"required": required, "optional": {slot: ("IMAGE",) for slot in SLOTS}}

    FUNCTION = "compose_fixed"
    CATEGORY = "Outfit Reference"

    @classmethod
    def _fixed_box(cls, box_name, width, height):
        return cls._normal_box_to_pixels(V2_FIXED_SLOTS[box_name], width, height)

    @classmethod
    def _bottom_waist_center(cls, cutout):
        """Find the garment's upper-body axis, ignoring asymmetric lower details.

        A full cutout can include side chains, asymmetric hems or a floor
        shadow. Those belong in the reference image but must not determine
        where the waistband sits relative to the outfit's shared axis.
        """
        alpha = np.asarray(cutout.getchannel("A")) >= 96
        minimum_span = max(3, int(round(cutout.width * 0.08)))
        for low, high in ((0.02, 0.12), (0.12, 0.45)):
            start = min(cutout.height - 1, max(0, int(round(cutout.height * low))))
            end = min(cutout.height, max(start + 1, int(round(cutout.height * high))))
            centers = []
            for row in alpha[start:end]:
                columns = np.flatnonzero(row)
                if columns.size >= minimum_span:
                    left, right = np.percentile(columns, (5, 95))
                    centers.append((left + right) / 2)
            if centers:
                return float(np.median(centers))
        return cutout.width / 2

    @classmethod
    def _fit_fixed_slot(cls, cutout, box_name, width, height, slot_padding):
        """Contain a tight cutout at the largest uniform scale inside its box."""
        hard = cls._fixed_box(box_name, width, height)
        box_width = max(1, hard[2] - hard[0])
        box_height = max(1, hard[3] - hard[1])
        requested_padding = max(0, int(slot_padding))
        padding = min(requested_padding, max(0, (min(box_width, box_height) - 1) // 2))
        inner_x0 = hard[0] + padding
        inner_y0 = hard[1] + padding
        inner_width = max(1, box_width - padding * 2)
        inner_height = max(1, box_height - padding * 2)
        scale = min(
            inner_width / max(1, cutout.width),
            inner_height / max(1, cutout.height),
        )
        waist_center = None
        axis_x = (hard[0] + hard[2]) / 2
        if box_name == "bottom":
            waist_center = cls._bottom_waist_center(cutout)
            # Keep the entire source in the safe box even when its waistband
            # is far from the cutout's geometric middle.
            scale = min(
                scale,
                (axis_x - inner_x0) / max(1, waist_center),
                (inner_x0 + inner_width - axis_x)
                / max(1, cutout.width - waist_center),
            )
        target_width = max(1, int(round(cutout.width * scale)))
        target_height = max(1, int(round(cutout.height * scale)))
        target_width = min(target_width, inner_width)
        target_height = min(target_height, inner_height)
        if waist_center is None:
            x = inner_x0 + (inner_width - target_width) // 2
        else:
            x = int(round(axis_x - waist_center * target_width / cutout.width))
            x = min(max(x, inner_x0), inner_x0 + inner_width - target_width)
        y = inner_y0 + (inner_height - target_height) // 2
        return (x, y, target_width, target_height), hard

    @staticmethod
    def _composite_flat(canvas, cutout, placement):
        """Alpha composite without V1's synthetic product shadow."""
        x, y, target_width, target_height = placement
        if min(cutout.width, cutout.height, target_width, target_height) >= 8:
            source_aspect = cutout.width / cutout.height
            target_aspect = target_width / target_height
            axis_change = max(source_aspect / target_aspect, target_aspect / source_aspect)
            if axis_change > 1.04:
                raise ValueError(
                    f"non-uniform resize is forbidden (axis ratio changed {axis_change:.3f}x)"
                )
        resized = cutout.resize((target_width, target_height), Image.Resampling.LANCZOS)
        canvas.alpha_composite(resized, (x, y))
        return (x, y, x + target_width, y + target_height)

    def _fixed_earrings(self, cutout, width, height, slot_padding):
        parts = self._split_pair(cutout)
        box_names = (
            ("earrings_left", "earrings_right")
            if len(parts) == 2
            else ("earrings_right",)
        )
        results = []
        for part, box_name in zip(parts, box_names):
            placement, hard = self._fit_fixed_slot(
                part, box_name, width, height, slot_padding
            )
            label = "earrings L" if box_name.endswith("left") else "earrings R"
            results.append((part, placement, hard, label))
        return results

    def _draw_fixed_guides(self, canvas, records, width, waist_marker=None):
        draw = ImageDraw.Draw(canvas)
        line_width = max(1, width // 512)
        red = (255, 54, 54, 255)
        cyan = (0, 155, 210, 255)
        centre_x = int(round(V2_OUTFIT_CENTER_X * width))
        self._dashed_line(
            draw,
            (centre_x, 0, centre_x, canvas.height - 1),
            (125, 125, 125, 210),
            width=1,
        )
        for hard, actual, label in records:
            draw.rectangle(hard, outline=red, width=line_width)
            draw.rectangle(actual, outline=cyan, width=line_width)
            draw.text((hard[0] + 4, hard[1] + 4), label, fill=red)
        if waist_marker is not None:
            x, y = waist_marker
            radius = max(4, width // 256)
            green = (0, 145, 80, 255)
            draw.line((x - radius, y, x + radius, y), fill=green, width=2)
            draw.line((x, y - radius, x, y + radius), fill=green, width=2)
            draw.text((x + radius + 3, y + 2), "waist anchor", fill=green)

    def compose_fixed(
        self,
        width,
        height,
        background,
        background_threshold,
        slot_padding,
        show_layout_guides=False,
        **images,
    ):
        colors = {
            "white": (255, 255, 255),
            "off_white": (248, 248, 245),
            "light_gray": (238, 238, 238),
        }
        canvas = Image.new("RGBA", (width, height), colors[background] + (255,))
        records = []
        cutouts = {}
        cutout_layout = {"background_threshold": background_threshold}
        waist_marker = None

        for slot in SLOTS:
            tensor = images.get(slot)
            if tensor is None:
                continue
            cutout = self._cutout(self._image(tensor), slot, cutout_layout)
            if cutout is None:
                raise ValueError(
                    f"{slot} image has no detectable foreground; check the image/background "
                    "or adjust background_threshold"
                )
            cutouts[slot] = cutout

        for slot in V2_DRAW_ORDER:
            cutout = cutouts.get(slot)
            if cutout is None:
                continue
            if slot == "earrings":
                for part, placement, hard, label in self._fixed_earrings(
                    cutout, width, height, slot_padding
                ):
                    actual = self._composite_flat(canvas, part, placement)
                    records.append((hard, actual, label))
                continue
            placement, hard = self._fit_fixed_slot(
                cutout, slot, width, height, slot_padding
            )
            actual = self._composite_flat(canvas, cutout, placement)
            records.append((hard, actual, slot))
            if slot == "bottom":
                waist_marker = (
                    int(round(actual[0] + self._bottom_waist_center(cutout) * (actual[2] - actual[0]) / cutout.width)),
                    int(round(actual[1] + (actual[3] - actual[1]) * 0.07)),
                )

        if show_layout_guides:
            self._draw_fixed_guides(canvas, records, width, waist_marker)
        output = torch.from_numpy(
            np.asarray(canvas.convert("RGB")).astype(np.float32) / 255.0
        ).unsqueeze(0)
        return (output,)


class OutfitReferenceComposerV3Top(OutfitReferenceComposerV2):
    """Compose upper-body products with semantic empty-row compaction."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "width": ("INT", {"default": 1536, "min": 256, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 2048, "min": 256, "max": 2048, "step": 64}),
            "background": (["off_white", "white", "light_gray"], {"default": "off_white"}),
            "background_threshold": (
                "INT",
                {"default": 24, "min": 4, "max": 100, "step": 1},
            ),
            "slot_padding": (
                "INT",
                {"default": 10, "min": 0, "max": 64, "step": 1},
            ),
            "show_layout_guides": ("BOOLEAN", {"default": False}),
        }
        return {
            "required": required,
            "optional": {slot: ("IMAGE",) for slot in V3_TOP_SLOTS},
        }

    FUNCTION = "compose_top"
    CATEGORY = "Outfit Reference"

    @classmethod
    def _fixed_box(cls, box_name, width, height):
        return cls._normal_box_to_pixels(V3_TOP_FIXED_SLOTS[box_name], width, height)

    def _draw_top_guides(self, canvas, records, width):
        draw = ImageDraw.Draw(canvas)
        line_width = max(1, width // 512)
        red = (255, 54, 54, 255)
        cyan = (0, 155, 210, 255)
        centre_x = int(round(V3_TOP_CENTER_X * width))
        self._dashed_line(
            draw,
            (centre_x, 0, centre_x, canvas.height - 1),
            (125, 125, 125, 210),
            width=1,
        )
        for hard, actual, label in records:
            draw.rectangle(hard, outline=red, width=line_width)
            draw.rectangle(actual, outline=cyan, width=line_width)
            draw.text((hard[0] + 4, hard[1] + 4), label, fill=red)

    @classmethod
    def _dynamic_boxes(cls, cutouts, width, height):
        """Return V3_TOP boxes after collapsing only empty semantic rows."""
        active = set(cutouts)
        row_names = set()
        if "hat" in active:
            row_names.add("hat")
        if active.intersection({"glasses", "earrings"}):
            row_names.add("face")
        if active.intersection({"top", "necklace"}):
            row_names.add("top")
        rows = _allocate_active_rows(V3_TOP_ROWS, row_names)
        boxes = dict(V3_TOP_FIXED_SLOTS)
        if "hat" in rows:
            boxes["hat"] = _remap_box_y(
                V3_TOP_FIXED_SLOTS["hat"], V3_TOP_ROWS["hat"], rows["hat"]
            )
        if "face" in rows:
            for name in ("glasses", "earrings_left", "earrings_right"):
                boxes[name] = _remap_box_y(
                    V3_TOP_FIXED_SLOTS[name], V3_TOP_ROWS["face"], rows["face"]
                )
        if "top" in rows:
            boxes["top"] = _remap_box_y(
                V3_TOP_FIXED_SLOTS["top"], V3_TOP_ROWS["top"], rows["top"]
            )
            boxes["necklace"] = _remap_box_y(
                V3_TOP_FIXED_SLOTS["necklace"],
                V3_TOP_ROWS["top"],
                rows["top"],
            )
        return {
            name: cls._normal_box_to_pixels(box, width, height)
            for name, box in boxes.items()
        }

    @classmethod
    def _fit_dynamic_slot(cls, cutout, box_name, hard_boxes, slot_padding):
        anchor = cls._bottom_waist_center(cutout) if box_name == "bottom" else None
        hard = hard_boxes[box_name]
        return (
            _fit_cutout_in_box(cutout, hard, slot_padding, anchor),
            hard,
        )

    def _dynamic_earrings(self, cutout, hard_boxes, slot_padding):
        parts = self._split_pair(cutout)
        box_names = (
            ("earrings_left", "earrings_right")
            if len(parts) == 2
            else ("earrings_right",)
        )
        results = []
        for part, box_name in zip(parts, box_names):
            placement, hard = self._fit_dynamic_slot(
                part, box_name, hard_boxes, slot_padding
            )
            label = "earrings L" if box_name.endswith("left") else "earrings R"
            results.append((part, placement, hard, label))
        return results

    def compose_top(
        self,
        width,
        height,
        background,
        background_threshold,
        slot_padding,
        show_layout_guides=False,
        **images,
    ):
        unsupported = sorted(
            slot
            for slot, tensor in images.items()
            if slot not in V3_TOP_SLOTS and tensor is not None
        )
        if unsupported:
            raise ValueError(
                "V3_TOP only accepts top, hat, glasses, earrings and necklace; "
                f"unsupported inputs: {', '.join(unsupported)}"
            )

        colors = {
            "white": (255, 255, 255),
            "off_white": (248, 248, 245),
            "light_gray": (238, 238, 238),
        }
        canvas = Image.new("RGBA", (width, height), colors[background] + (255,))
        occupancy = Image.new("L", (width, height), 0)
        records = []
        cutouts = {}
        cutout_layout = {"background_threshold": background_threshold}

        for slot in V3_TOP_SLOTS:
            tensor = images.get(slot)
            if tensor is None:
                continue
            cutout = self._cutout(self._image(tensor), slot, cutout_layout)
            if cutout is None:
                raise ValueError(
                    f"{slot} image has no detectable foreground; check the image/background "
                    "or adjust background_threshold"
                )
            cutouts[slot] = cutout

        hard_boxes = self._dynamic_boxes(cutouts, width, height)
        complete_set = all(slot in cutouts for slot in V3_TOP_SLOTS)

        for slot in V3_TOP_DRAW_ORDER:
            cutout = cutouts.get(slot)
            if cutout is None:
                continue
            if slot == "earrings":
                for part, placement, hard, label in self._dynamic_earrings(
                    cutout, hard_boxes, slot_padding
                ):
                    actual = self._composite_flat(canvas, part, placement)
                    _add_cutout_to_occupancy(occupancy, part, placement)
                    records.append((hard, actual, label))
                continue
            placement, hard = self._fit_dynamic_slot(
                cutout, slot, hard_boxes, slot_padding
            )
            actual = self._composite_flat(canvas, cutout, placement)
            _add_cutout_to_occupancy(occupancy, cutout, placement)
            records.append((hard, actual, slot))

        if show_layout_guides:
            self._draw_top_guides(canvas, records, width)
        output = _canvas_to_rgb_tensor(
            canvas,
            occupancy,
            # Guide mode is a layout diagnostic: retain the requested canvas
            # so coordinates and empty-row outlines remain inspectable.
            crop=not complete_set and not show_layout_guides,
            slot_padding=slot_padding,
        )
        return (output,)


class OutfitReferenceComposerV3(OutfitReferenceComposerV2):
    """Full-body semantic layout with two description-driven overlap modes.

    V3 does not infer semantics from pixels.  Wardrobe Library supplies the
    selected items' layout tags as a tiny JSON object, for example
    ``{"top":"top_coverwaist","bottom":"bottom_shoehalf"}``.  Missing semantic
    rows collapse locally; a complete untagged set stays on the exact V2 path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        optional = dict(inputs["optional"])
        optional["layout_tags_json"] = (
            "STRING",
            {"forceInput": True, "default": "{}", "multiline": False},
        )
        return {"required": dict(inputs["required"]), "optional": optional}

    FUNCTION = "compose_v3"
    CATEGORY = "Outfit Reference"

    @staticmethod
    def _parse_layout_tags(layout_tags_json):
        if layout_tags_json is None or not str(layout_tags_json).strip():
            return {}
        try:
            payload = json.loads(str(layout_tags_json))
        except json.JSONDecodeError as exc:
            raise ValueError(f"layout_tags_json is not valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("layout_tags_json must be a JSON object")

        tags = {}
        for slot, raw_tag in payload.items():
            if slot not in V3_LAYOUT_TAGS:
                raise ValueError(
                    "layout_tags_json only accepts top and bottom keys; "
                    f"received {slot!r}"
                )
            tag = str(raw_tag or "").strip().lower()
            if not tag:
                continue
            expected = V3_LAYOUT_TAGS[slot]
            if tag != expected:
                raise ValueError(
                    f"layout_tags_json.{slot} must be {expected!r}; received {tag!r}"
                )
            tags[slot] = tag
        return tags

    @staticmethod
    def _placement_box(placement):
        x, y, target_width, target_height = placement
        return (x, y, x + target_width, y + target_height)

    @staticmethod
    def _union_boxes(first, second):
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    @staticmethod
    def _visible_alpha_bbox(cutout, threshold=8):
        """Return the alpha-visible box using the same cutoff as ``_cutout``."""
        return cutout.getchannel("A").point(
            lambda value: 255 if value >= threshold else 0
        ).getbbox()

    @classmethod
    def _dynamic_boxes(cls, cutouts, width, height):
        """Collapse empty head/body rows without changing full-set geometry."""
        active = set(cutouts)
        head_names = set()
        if "hat" in active:
            head_names.add("hat")
        if active.intersection({"glasses", "earrings"}):
            head_names.add("face")
        body_names = set()
        if active.intersection({"top", "necklace"}):
            body_names.add("top")
        if "bottom" in active:
            body_names.add("bottom")
        if active.intersection({"socks", "shoes"}):
            body_names.add("foot")

        head_rows = _allocate_active_rows(V3_FULL_HEAD_ROWS, head_names)
        body_rows = _allocate_active_rows(V3_FULL_BODY_ROWS, body_names)
        boxes = dict(V2_FIXED_SLOTS)
        if "hat" in head_rows:
            boxes["hat"] = _remap_box_y(
                V2_FIXED_SLOTS["hat"],
                V3_FULL_HEAD_ROWS["hat"],
                head_rows["hat"],
            )
        if "face" in head_rows:
            for name in ("glasses", "earrings_left", "earrings_right"):
                boxes[name] = _remap_box_y(
                    V2_FIXED_SLOTS[name],
                    V3_FULL_HEAD_ROWS["face"],
                    head_rows["face"],
                )
        if "top" in body_rows:
            boxes["top"] = _remap_box_y(
                V2_FIXED_SLOTS["top"],
                V3_FULL_BODY_ROWS["top"],
                body_rows["top"],
            )
            boxes["necklace"] = _remap_box_y(
                V2_FIXED_SLOTS["necklace"],
                V3_FULL_BODY_ROWS["top"],
                body_rows["top"],
            )
        if "bottom" in body_rows:
            boxes["bottom"] = _remap_box_y(
                V2_FIXED_SLOTS["bottom"],
                V3_FULL_BODY_ROWS["bottom"],
                body_rows["bottom"],
            )
        if "foot" in body_rows:
            for name in ("socks", "shoes"):
                boxes[name] = _remap_box_y(
                    V2_FIXED_SLOTS[name],
                    V3_FULL_BODY_ROWS["foot"],
                    body_rows["foot"],
                )
        return {
            name: cls._normal_box_to_pixels(box, width, height)
            for name, box in boxes.items()
        }

    @classmethod
    def _fit_dynamic_slot(cls, cutout, box_name, hard_boxes, slot_padding):
        anchor = cls._bottom_waist_center(cutout) if box_name == "bottom" else None
        hard = hard_boxes[box_name]
        return (
            _fit_cutout_in_box(cutout, hard, slot_padding, anchor),
            hard,
        )

    def _dynamic_earrings(self, cutout, hard_boxes, slot_padding):
        parts = self._split_pair(cutout)
        box_names = (
            ("earrings_left", "earrings_right")
            if len(parts) == 2
            else ("earrings_right",)
        )
        results = []
        for part, box_name in zip(parts, box_names):
            placement, hard = self._fit_dynamic_slot(
                part, box_name, hard_boxes, slot_padding
            )
            label = "earrings L" if box_name.endswith("left") else "earrings R"
            results.append((part, placement, hard, label))
        return results

    @staticmethod
    def _scale_to_lower_anchor(cutout, base_placement, target_bottom, axis_x, source_axis_x):
        """Uniformly scale from a fixed upper edge to an object-relative line."""
        _base_x, y, _base_width, _base_height = base_placement
        target_bottom = int(round(target_bottom))
        target_height = max(1, target_bottom - y)
        scale = target_height / max(1, cutout.height)
        target_width = max(1, int(round(cutout.width * scale)))
        # Use the realised horizontal scale so the selected source axis remains
        # on the common outfit axis even after integer rounding.
        x = int(round(axis_x - source_axis_x * target_width / cutout.width))
        return (x, y, target_width, target_height)

    def _v3_relative_placements(
        self,
        cutouts,
        width,
        height,
        slot_padding,
        top_coverwaist,
        bottom_shoehalf,
        dynamic_hard_boxes=None,
    ):
        """Start with active V3 row placements, then meet semantic boundaries.

        The upper edge of each tagged garment remains at its dynamically
        allocated row. Scaling is uniform and anchored to the shared outfit
        axis. This makes the result independent of source aspect ratio and
        avoids claiming overlap merely because a larger hard box was allocated.
        """
        placements = {}
        hard_boxes = {}
        for slot, cutout in cutouts.items():
            if slot == "earrings":
                continue
            if dynamic_hard_boxes is None:
                placement, hard = self._fit_fixed_slot(
                    cutout, slot, width, height, slot_padding
                )
            else:
                placement, hard = self._fit_dynamic_slot(
                    cutout, slot, dynamic_hard_boxes, slot_padding
                )
            placements[slot] = placement
            hard_boxes[slot] = hard

        axis_x = V2_OUTFIT_CENTER_X * width
        if bottom_shoehalf:
            shoe_placement = placements["shoes"]
            shoe_top = shoe_placement[1]
            shoe_height = shoe_placement[3]
            target_hem = shoe_top + int(
                round(shoe_height * V3_BOTTOM_SHOEHALF_COVER_FRACTION)
            )
            seam_overlap = max(
                1,
                int(
                    round(
                        shoe_height * V3_BOTTOM_SHOEHALF_SEAM_OVERLAP_FRACTION
                    )
                ),
            )
            visible_target_hem = target_hem + seam_overlap
            bottom_cutout = cutouts["bottom"]
            # ``_cutout`` deliberately leaves a small transparent safety pad.
            # Anchor the *visible alpha hem* to the semantic two-thirds line,
            # rather than anchoring the padded image rectangle there.  This
            # prevents the canvas from appearing as a white seam while keeping
            # the shoe crop strictly limited to its lower third.
            visible_bbox = self._visible_alpha_bbox(bottom_cutout)
            visible_hem = visible_bbox[3] if visible_bbox else bottom_cutout.height
            base_y = placements["bottom"][1]
            padded_target_bottom = base_y + int(
                np.ceil(
                    (visible_target_hem - base_y)
                    * bottom_cutout.height
                    / max(1, visible_hem)
                )
            )
            placements["bottom"] = self._scale_to_lower_anchor(
                bottom_cutout,
                placements["bottom"],
                padded_target_bottom,
                axis_x,
                self._bottom_waist_center(bottom_cutout),
            )
            hard_boxes["bottom"] = self._union_boxes(
                hard_boxes["bottom"],
                self._placement_box(placements["bottom"]),
            )

        if top_coverwaist:
            bottom_placement = placements["bottom"]
            hip_y = bottom_placement[1] + int(
                round(bottom_placement[3] * V3_TOP_COVERWAIST_HIP_FRACTION)
            )
            top_cutout = cutouts["top"]
            placements["top"] = self._scale_to_lower_anchor(
                top_cutout,
                placements["top"],
                hip_y,
                axis_x,
                top_cutout.width / 2,
            )
            hard_boxes["top"] = self._union_boxes(
                hard_boxes["top"],
                self._placement_box(placements["top"]),
            )

        return placements, hard_boxes

    @staticmethod
    def _shoe_toe_third(cutout, placement):
        """Return only the lower shoe third at its original placed scale.

        Cropping the source and placement together (rather than hiding it
        behind a narrow trouser cutout) guarantees no shoe upper can leak at
        the sides above the semantic two-thirds coverage line.
        """
        source_start = int(
            round(cutout.height * V3_BOTTOM_SHOEHALF_COVER_FRACTION)
        )
        source_start = min(max(0, source_start), cutout.height - 1)
        x, y, target_width, target_height = placement
        target_start = int(
            round(target_height * V3_BOTTOM_SHOEHALF_COVER_FRACTION)
        )
        target_start = min(max(0, target_start), target_height - 1)
        toe = cutout.crop((0, source_start, cutout.width, cutout.height))
        toe_placement = (
            x,
            y + target_start,
            target_width,
            target_height - target_start,
        )
        return toe, toe_placement

    @staticmethod
    def _v3_draw_order(top_coverwaist, bottom_shoehalf):
        if top_coverwaist and bottom_shoehalf:
            main = ("socks", "shoes", "bottom", "top")
        elif top_coverwaist:
            main = ("bottom", "top", "socks", "shoes")
        elif bottom_shoehalf:
            main = ("top", "socks", "shoes", "bottom")
        else:
            main = V2_DRAW_ORDER[:4]
        return main + V2_DRAW_ORDER[4:]

    def compose_v3(
        self,
        width,
        height,
        background,
        background_threshold,
        slot_padding,
        show_layout_guides=False,
        layout_tags_json="{}",
        **images,
    ):
        tags = self._parse_layout_tags(layout_tags_json)
        top_coverwaist = (
            tags.get("top") == "top_coverwaist"
            and images.get("top") is not None
            and images.get("bottom") is not None
        )
        bottom_shoehalf = (
            tags.get("bottom") == "bottom_shoehalf"
            and images.get("bottom") is not None
            and images.get("shoes") is not None
        )

        complete_set = all(images.get(slot) is not None for slot in SLOTS)
        # A complete, untagged outfit remains byte-for-byte on the proven V2
        # path.  Dynamic compaction is only observable when a semantic row is
        # actually empty.
        if complete_set and not top_coverwaist and not bottom_shoehalf:
            return self.compose_fixed(
                width,
                height,
                background,
                background_threshold,
                slot_padding,
                show_layout_guides,
                **images,
            )

        colors = {
            "white": (255, 255, 255),
            "off_white": (248, 248, 245),
            "light_gray": (238, 238, 238),
        }
        canvas = Image.new("RGBA", (width, height), colors[background] + (255,))
        occupancy = Image.new("L", (width, height), 0)
        records = []
        cutouts = {}
        cutout_layout = {"background_threshold": background_threshold}
        waist_marker = None

        for slot in SLOTS:
            tensor = images.get(slot)
            if tensor is None:
                continue
            cutout = self._cutout(self._image(tensor), slot, cutout_layout)
            if cutout is None:
                raise ValueError(
                    f"{slot} image has no detectable foreground; check the image/background "
                    "or adjust background_threshold"
                )
            cutouts[slot] = cutout

        dynamic_hard_boxes = self._dynamic_boxes(cutouts, width, height)

        placements, hard_boxes = self._v3_relative_placements(
            cutouts,
            width,
            height,
            slot_padding,
            top_coverwaist,
            bottom_shoehalf,
            dynamic_hard_boxes,
        )

        for slot in self._v3_draw_order(top_coverwaist, bottom_shoehalf):
            cutout = cutouts.get(slot)
            if cutout is None:
                continue
            if slot == "earrings":
                for part, placement, hard, label in self._dynamic_earrings(
                    cutout, dynamic_hard_boxes, slot_padding
                ):
                    actual = self._composite_flat(canvas, part, placement)
                    _add_cutout_to_occupancy(occupancy, part, placement)
                    records.append((hard, actual, label))
                continue
            placement = placements[slot]
            hard = hard_boxes[slot]
            render_cutout = cutout
            label = slot
            if slot == "shoes" and bottom_shoehalf:
                render_cutout, placement = self._shoe_toe_third(cutout, placement)
                label += " / toe 1/3"
            elif slot == "top" and top_coverwaist:
                label += " / top_coverwaist"
            elif slot == "bottom" and bottom_shoehalf:
                label += " / bottom_shoehalf"
            actual = self._composite_flat(canvas, render_cutout, placement)
            _add_cutout_to_occupancy(occupancy, render_cutout, placement)
            records.append((hard, actual, label))
            if slot == "bottom":
                waist_marker = (
                    int(
                        round(
                            actual[0]
                            + self._bottom_waist_center(cutout)
                            * (actual[2] - actual[0])
                            / cutout.width
                        )
                    ),
                    int(round(actual[1] + (actual[3] - actual[1]) * 0.07)),
                )

        if show_layout_guides:
            self._draw_fixed_guides(canvas, records, width, waist_marker)
        output = _canvas_to_rgb_tensor(
            canvas,
            occupancy,
            # Guide mode intentionally keeps the full requested canvas; the
            # production render crops only when a semantic slot is absent.
            crop=not complete_set and not show_layout_guides,
            slot_padding=slot_padding,
        )
        return (output,)


NODE_CLASS_MAPPINGS = {
    "OutfitReferenceComposer": OutfitReferenceComposer,
    "OutfitReferenceComposerV2": OutfitReferenceComposerV2,
    "OutfitReferenceComposerV3": OutfitReferenceComposerV3,
    "OutfitReferenceComposerV3Top": OutfitReferenceComposerV3Top,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OutfitReferenceComposer": "Outfit Reference Composer",
    "OutfitReferenceComposerV2": "Outfit Reference Composer V2 (Fixed Slots)",
    "OutfitReferenceComposerV3": "Outfit Reference Composer V3 (Tagged Overlap)",
    "OutfitReferenceComposerV3Top": "Outfit Reference Composer V3_TOP (Upper Body Only)",
}
