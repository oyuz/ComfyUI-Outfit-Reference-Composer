import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    OutfitReferenceComposerV2,
    SLOTS,
    V2_DRAW_ORDER,
    V2_FIXED_SLOTS,
    V2_OUTFIT_CENTER_X,
)


def rgba_tensor(image):
    array = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def axis_change(source, placement):
    source_aspect = source.width / source.height
    target_aspect = placement[2] / placement[3]
    return max(source_aspect / target_aspect, target_aspect / source_aspect)


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class OutfitReferenceComposerV2Tests(unittest.TestCase):
    def setUp(self):
        self.node = OutfitReferenceComposerV2()

    def test_v1_and_v2_are_registered(self):
        self.assertIn("OutfitReferenceComposer", NODE_CLASS_MAPPINGS)
        self.assertIn("OutfitReferenceComposerV2", NODE_CLASS_MAPPINGS)

    def test_input_socket_is_the_only_item_semantic(self):
        inputs = self.node.INPUT_TYPES()
        self.assertNotIn("outfit_spec", inputs["required"])
        self.assertEqual(list(inputs["optional"]), list(SLOTS))
        self.assertLess(list(inputs["optional"]).index("socks"), list(inputs["optional"]).index("shoes"))

    def test_fixed_boxes_only_allow_necklace_to_overlap_top(self):
        allowed_overlap = frozenset(("top", "necklace"))
        for left, right in combinations(V2_FIXED_SLOTS, 2):
            with self.subTest(left=left, right=right):
                pair = frozenset((left, right))
                self.assertEqual(
                    overlaps(V2_FIXED_SLOTS[left], V2_FIXED_SLOTS[right]),
                    pair == allowed_overlap,
                )

    def test_necklace_overlays_upper_top_and_renders_after_it(self):
        necklace = V2_FIXED_SLOTS["necklace"]
        top = V2_FIXED_SLOTS["top"]
        self.assertTrue(overlaps(necklace, top))
        self.assertLessEqual(necklace[3], top[1] + 0.35 * (top[3] - top[1]))
        self.assertGreater(
            V2_DRAW_ORDER.index("necklace"),
            V2_DRAW_ORDER.index("top"),
        )

    def test_necklace_top_bottom_and_shoes_share_one_center_axis(self):
        for slot in ("necklace", "top", "bottom", "shoes"):
            with self.subTest(slot=slot):
                box = V2_FIXED_SLOTS[slot]
                self.assertAlmostEqual((box[0] + box[2]) / 2, V2_OUTFIT_CENTER_X)

        for width, height in ((768, 1024), (1536, 2048)):
            for slot, size in (
                ("necklace", (300, 180)),
                ("top", (500, 700)),
                ("bottom", (350, 850)),
                ("shoes", (450, 220)),
            ):
                with self.subTest(slot=slot, width=width):
                    cutout = Image.new("RGBA", size, (80, 50, 30, 255))
                    actual, hard = self.node._fit_fixed_slot(
                        cutout, slot, width, height, 10
                    )
                    self.assertLessEqual(
                        abs(actual[0] + actual[2] / 2 - V2_OUTFIT_CENTER_X * width),
                        1.0,
                    )
                    self.assertLessEqual(
                        abs((hard[0] + hard[2]) / 2 - V2_OUTFIT_CENTER_X * width),
                        1.0,
                    )
                    self.assertLessEqual(axis_change(cutout, actual), 1.02)

    def test_asymmetric_bottom_aligns_waist_not_full_cutout_box(self):
        cutout = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
        draw = ImageDraw.Draw(cutout)
        draw.rectangle((140, 0, 260, 70), fill=(30, 30, 30, 255))
        draw.polygon([(140, 70), (260, 70), (290, 399), (20, 399)], fill=(30, 30, 30, 255))
        anchor = self.node._bottom_waist_center(cutout)
        self.assertAlmostEqual(anchor, 200, delta=1)

        for width, height in ((768, 1024), (1536, 2048)):
            with self.subTest(width=width):
                actual, hard = self.node._fit_fixed_slot(cutout, "bottom", width, height, 10)
                axis = (hard[0] + hard[2]) / 2
                scaled_anchor = actual[0] + anchor * actual[2] / cutout.width
                self.assertAlmostEqual(scaled_anchor, axis, delta=1.5)
                self.assertGreater(abs(actual[0] + actual[2] / 2 - axis), 10)
                self.assertGreaterEqual(actual[0], hard[0] + 10)
                self.assertLessEqual(actual[0] + actual[2], hard[2] - 10)
                self.assertLessEqual(axis_change(cutout, actual), 1.02)

        guided = self.node.compose_fixed(
            768, 1024, "white", 24, 10, True, bottom=rgba_tensor(cutout)
        )[0]
        clean = self.node.compose_fixed(
            768, 1024, "white", 24, 10, False, bottom=rgba_tensor(cutout)
        )[0]
        axis = int(round(V2_OUTFIT_CENTER_X * 768))
        guided_pixels = np.rint(guided[0].numpy() * 255).astype(np.uint8)
        clean_pixels = np.rint(clean[0].numpy() * 255).astype(np.uint8)
        green_on_axis = (
            (guided_pixels[:, axis, 0] < 20)
            & (guided_pixels[:, axis, 1] > 120)
            & (guided_pixels[:, axis, 2] < 100)
        )
        self.assertTrue(np.any(green_on_axis))
        self.assertFalse(np.any(np.all(clean_pixels == (0, 145, 80), axis=2)))

    def test_head_slots_are_larger_and_compact(self):
        hat = V2_FIXED_SLOTS["hat"]
        glasses = V2_FIXED_SLOTS["glasses"]
        top = V2_FIXED_SLOTS["top"]
        self.assertGreaterEqual(hat[2] - hat[0], 0.30)
        self.assertGreaterEqual(hat[3] - hat[1], 0.095)
        self.assertGreaterEqual(glasses[2] - glasses[0] + 1e-9, 0.28)
        self.assertGreaterEqual(glasses[3] - glasses[1] + 1e-9, 0.065)
        self.assertGreaterEqual(glasses[1] - hat[3], 0.0)
        self.assertLessEqual(glasses[1] - hat[3], 0.003)
        self.assertGreaterEqual(top[1] - glasses[3], 0.0)
        self.assertLessEqual(top[1] - glasses[3], 0.01)

    def test_socks_sit_left_of_taller_narrower_footwear_box(self):
        bottom = V2_FIXED_SLOTS["bottom"]
        socks = V2_FIXED_SLOTS["socks"]
        shoes = V2_FIXED_SLOTS["shoes"]
        self.assertGreaterEqual(socks[1] - bottom[3], 0.0)
        self.assertLessEqual(socks[1] - bottom[3], 0.01)
        self.assertGreaterEqual(shoes[1] - bottom[3], 0.0)
        self.assertLessEqual(shoes[1] - bottom[3], 0.01)
        self.assertLessEqual(socks[2], shoes[0])
        self.assertEqual((socks[1], socks[3]), (shoes[1], shoes[3]))
        self.assertLessEqual(shoes[2] - shoes[0], 0.40)
        self.assertGreaterEqual(shoes[3] - shoes[1] + 1e-9, 0.17)
        self.assertGreaterEqual(socks[3] - socks[1] + 1e-9, 0.15)

    def test_tall_boot_uses_full_footwear_height_without_distortion(self):
        boot = Image.new("RGBA", (240, 600), (80, 50, 30, 255))
        placement, hard = self.node._fit_fixed_slot(
            boot, "shoes", 1536, 2048, 10
        )
        self.assertLessEqual(axis_change(boot, placement), 1.02)
        self.assertGreaterEqual(placement[0], hard[0] + 10)
        self.assertLessEqual(placement[0] + placement[2], hard[2] - 10)
        self.assertEqual(placement[3], hard[3] - hard[1] - 20)

    def test_cutout_is_maximised_without_distortion(self):
        for size in ((640, 420), (260, 700), (1000, 100)):
            with self.subTest(size=size):
                cutout = Image.new("RGBA", size, (70, 80, 120, 255))
                placement, hard = self.node._fit_fixed_slot(
                    cutout, "top", 1536, 2048, 10
                )
                self.assertLessEqual(axis_change(cutout, placement), 1.02)
                self.assertGreaterEqual(placement[0], hard[0] + 10)
                self.assertGreaterEqual(placement[1], hard[1] + 10)
                self.assertLessEqual(placement[0] + placement[2], hard[2] - 10)
                self.assertLessEqual(placement[1] + placement[3], hard[3] - 10)
                inner_width = hard[2] - hard[0] - 20
                inner_height = hard[3] - hard[1] - 20
                self.assertTrue(
                    abs(placement[2] - inner_width) <= 1
                    or abs(placement[3] - inner_height) <= 1
                )

    def test_slot_position_does_not_depend_on_other_inputs(self):
        top = Image.new("RGBA", (600, 500), (80, 30, 30, 255))
        alone = self.node._fit_fixed_slot(top, "top", 1536, 2048, 10)
        with_everything_else = self.node._fit_fixed_slot(top, "top", 1536, 2048, 10)
        self.assertEqual(alone, with_everything_else)

    def test_pair_earrings_use_two_fixed_boxes(self):
        pair = Image.new("RGBA", (240, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pair)
        draw.ellipse((10, 10, 90, 90), fill=(90, 70, 40, 255))
        draw.ellipse((150, 10, 230, 90), fill=(90, 70, 40, 255))
        placements = self.node._fixed_earrings(pair, 1536, 2048, 10)
        self.assertEqual(len(placements), 2)
        self.assertEqual([entry[3] for entry in placements], ["earrings L", "earrings R"])

    def test_single_earring_uses_deterministic_right_box(self):
        single = Image.new("RGBA", (80, 100), (90, 70, 40, 255))
        placements = self.node._fixed_earrings(single, 1536, 2048, 10)
        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0][3], "earrings R")

    def test_blank_white_input_reports_foreground_error(self):
        blank = rgba_tensor(Image.new("RGB", (256, 256), "white"))
        with self.assertRaisesRegex(ValueError, "no detectable foreground"):
            self.node.compose_fixed(
                768, 1024, "white", 24, 8, False, top=blank
            )

    def test_all_slots_compose_to_requested_size(self):
        images = {}
        for index, slot in enumerate(SLOTS):
            product = Image.new("RGBA", (260 + index * 7, 220 + index * 11), (0, 0, 0, 0))
            ImageDraw.Draw(product).rounded_rectangle(
                (18, 16, product.width - 18, product.height - 16),
                radius=18,
                fill=(40 + index * 12, 80, 150, 255),
            )
            images[slot] = rgba_tensor(product)
        output = self.node.compose_fixed(
            1536, 2048, "off_white", 24, 10, False, **images
        )[0]
        self.assertEqual(tuple(output.shape), (1, 2048, 1536, 3))

    def test_image_batches_remain_rejected(self):
        image = rgba_tensor(Image.new("RGBA", (100, 100), (80, 80, 80, 255)))
        batch = torch.cat((image, image), dim=0)
        with self.assertRaisesRegex(ValueError, "exactly one IMAGE"):
            self.node.compose_fixed(
                768, 1024, "white", 24, 8, False, top=batch
            )


if __name__ == "__main__":
    unittest.main()
