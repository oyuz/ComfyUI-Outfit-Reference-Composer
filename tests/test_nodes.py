import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import OutfitReferenceComposer, SLOTS, TEMPLATE  # noqa: E402


def image_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class OutfitReferenceComposerTests(unittest.TestCase):
    def setUp(self):
        self.node = OutfitReferenceComposer()

    def test_node_has_socks_input_and_one_output(self):
        self.assertIn("socks", SLOTS)
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(self.node.RETURN_NAMES, ("outfit_reference",))

    def test_cutout_ignores_disconnected_edge_watermark(self):
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((150, 100, 360, 410), fill=(50, 70, 95))
        draw.text((2, 480), "watermark", fill=(215, 215, 215))
        cutout = self.node._cutout(image.convert("RGBA"), "top", {})
        self.assertIsNotNone(cutout)
        self.assertLess(cutout.width, 260)
        self.assertLess(cutout.height, 360)

    def test_same_fit_uses_same_effective_shoulder_width(self):
        short_top = Image.new("RGBA", (680, 480), (80, 20, 20, 255))
        long_top = Image.new("RGBA", (530, 650), (30, 80, 130, 255))
        short_placement, _, _ = self.node._top_placement(
            short_top,
            {"garment_type": "jersey", "sleeve_length": "short", "fit": "loose", "length": "crop"},
            768,
            1024,
        )
        long_placement, _, _ = self.node._top_placement(
            long_top,
            {"garment_type": "shirt", "sleeve_length": "long", "fit": "loose", "length": "hip"},
            768,
            1024,
        )
        short_shoulder = short_placement[2] * 0.68
        long_shoulder = long_placement[2] * 0.65
        self.assertLess(abs(short_shoulder - long_shoulder), 1)

    def test_top_neck_and_hem_are_hard_vertical_anchors(self):
        cutout = Image.new("RGBA", (530, 650), (30, 80, 130, 255))
        placement, _, _ = self.node._top_placement(
            cutout,
            {"garment_type": "shirt", "sleeve_length": "long", "fit": "loose", "length": "hip"},
            768,
            1024,
        )
        expected_top = round(TEMPLATE["landmarks"]["neck_y"] * 1024)
        expected_bottom = round(TEMPLATE["landmarks"]["top_hem_y"]["hip"] * 1024)
        self.assertLessEqual(abs(placement[1] - expected_top), 1)
        self.assertLessEqual(abs(placement[1] + placement[3] - expected_bottom), 1)

    def test_bottom_waist_and_hem_are_hard_vertical_anchors(self):
        cutout = Image.new("RGBA", (300, 600), (80, 80, 80, 255))
        placement, _, _ = self.node._bottom_placement(
            cutout,
            {"fit": "oversized", "waist": "low", "length": "floor", "silhouette": "baggy"},
            768,
            1024,
        )
        expected_top = round(TEMPLATE["landmarks"]["bottom_waist_y"]["low"] * 1024)
        expected_bottom = round(TEMPLATE["landmarks"]["bottom_hem_y"]["floor"] * 1024)
        self.assertLessEqual(abs(placement[1] - expected_top), 1)
        self.assertLessEqual(abs(placement[1] + placement[3] - expected_bottom), 1)

    def test_same_bottom_metadata_ignores_product_shot_aspect(self):
        narrow = Image.new("RGBA", (360, 400), (80, 80, 80, 255))
        wide = Image.new("RGBA", (620, 420), (80, 80, 80, 255))
        layout = {"fit": "loose", "waist": "mid", "length": "knee", "silhouette": "wide"}
        narrow_placement, _, _ = self.node._bottom_placement(narrow, layout, 768, 1024)
        wide_placement, _, _ = self.node._bottom_placement(wide, layout, 768, 1024)
        self.assertEqual(narrow_placement[2:], wide_placement[2:])

    def test_extreme_axis_distortion_is_rejected(self):
        extreme = Image.new("RGBA", (1000, 100), (80, 80, 80, 255))
        with self.assertRaisesRegex(ValueError, "axis distortion"):
            self.node._top_placement(
                extreme,
                {"garment_type": "shirt", "fit": "regular", "length": "hip"},
                768,
                1024,
            )

    def test_main_slot_boxes_do_not_overlap(self):
        boxes = TEMPLATE["hard_boxes"]
        self.assertLessEqual(boxes["top"][3], boxes["bottom"][1])
        self.assertLessEqual(boxes["bottom"][3], boxes["shoes"][1])

    def test_earring_boxes_are_above_top_and_clear_of_glasses(self):
        boxes = TEMPLATE["hard_boxes"]
        for side in ("earrings_left", "earrings_right"):
            self.assertLessEqual(boxes[side][3], boxes["top"][1])
            self.assertTrue(
                boxes[side][2] <= boxes["glasses"][0]
                or boxes[side][0] >= boxes["glasses"][2]
            )

    def test_paired_earrings_are_split_left_and_right(self):
        pair = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pair)
        draw.ellipse((15, 10, 75, 90), fill=(180, 120, 30, 255))
        draw.ellipse((125, 10, 185, 90), fill=(180, 120, 30, 255))
        placements = self.node._earrings_placements(pair, {"size": "medium"}, 768, 1024)
        self.assertEqual(len(placements), 2)
        self.assertLess(placements[0][1][0], placements[1][1][0])

    def test_single_earring_is_not_cut_in_half(self):
        single = Image.new("RGBA", (120, 100), (0, 0, 0, 0))
        ImageDraw.Draw(single).ellipse((20, 8, 100, 92), fill=(180, 120, 30, 255))
        placements = self.node._earrings_placements(single, {"size": "medium"}, 768, 1024)
        self.assertEqual(len(placements), 1)
        glasses_right = round(TEMPLATE["hard_boxes"]["glasses"][2] * 768)
        self.assertGreaterEqual(placements[0][1][0], glasses_right)

    def test_long_bottom_with_socks_is_rejected(self):
        cutout = Image.new("RGBA", (200, 350), (230, 230, 230, 255))
        with self.assertRaisesRegex(ValueError, "calf/ankle/floor"):
            self.node._socks_placement(
                cutout,
                {"length": "mid_calf"},
                {"garment_type": "pants", "length": "floor"},
                768,
                1024,
                True,
            )

    def test_long_skirt_with_socks_is_rejected(self):
        cutout = Image.new("RGBA", (200, 350), (230, 230, 230, 255))
        with self.assertRaisesRegex(ValueError, "calf/ankle/floor"):
            self.node._socks_placement(
                cutout,
                {"length": "mid_calf"},
                {"garment_type": "skirt", "length": "ankle"},
                768,
                1024,
                True,
            )

    def test_connected_bottom_without_metadata_rejects_socks(self):
        cutout = Image.new("RGBA", (200, 350), (230, 230, 230, 255))
        with self.assertRaisesRegex(ValueError, "calf/ankle/floor"):
            self.node._socks_placement(cutout, {"length": "crew"}, {}, 768, 1024, True)

    def test_short_bottom_and_socks_hard_boxes_do_not_overlap(self):
        bottom = Image.new("RGBA", (360, 250), (80, 80, 80, 255))
        socks = Image.new("RGBA", (220, 300), (230, 230, 230, 255))
        _, bottom_hard, _ = self.node._bottom_placement(
            bottom,
            {"fit": "loose", "waist": "low", "length": "knee", "silhouette": "wide"},
            768,
            1024,
        )
        _, socks_hard, _ = self.node._socks_placement(
            socks,
            {"length": "mid_calf"},
            {"garment_type": "shorts", "length": "knee"},
            768,
            1024,
            True,
        )
        self.assertLessEqual(bottom_hard[3], socks_hard[1])

    def test_compose_accepts_direct_item_fields(self):
        product = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(product).rectangle((70, 40, 186, 220), fill=(25, 25, 25))
        spec = {
            "items": {
                "top": {
                    "item_id": "test-top",
                    "slot": "top",
                    "garment_type": "tank",
                    "sleeve_length": "sleeveless",
                    "fit": "regular",
                    "length": "waist",
                }
            }
        }
        output = self.node.compose(
            json.dumps(spec),
            768,
            1024,
            "white",
            False,
            top=image_tensor(product),
        )[0]
        self.assertEqual(tuple(output.shape), (1, 1024, 768, 3))

    def test_parse_rejects_non_object_root(self):
        for value in ("null", "[]", '"top"'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "root must be"):
                self.node._parse(value)

    def test_unknown_enum_value_is_rejected(self):
        cutout = Image.new("RGBA", (400, 400), (80, 80, 80, 255))
        with self.assertRaisesRegex(ValueError, "top-test.fit"):
            self.node._top_placement(
                cutout,
                {"item_id": "top-test", "fit": "loos", "length": "waist"},
                768,
                1024,
            )

    def test_coverage_has_explicit_precedence_over_shorter_length(self):
        cutout = Image.new("RGBA", (400, 400), (80, 80, 80, 255))
        crop, _, crop_label = self.node._top_placement(
            cutout,
            {"fit": "regular", "length": "hip", "coverage": "show_midriff"},
            768,
            1024,
        )
        hip, _, hip_label = self.node._top_placement(
            cutout,
            {"fit": "regular", "length": "crop", "coverage": "cover_hip"},
            768,
            1024,
        )
        self.assertIn("/crop", crop_label)
        self.assertIn("/hip", hip_label)
        self.assertLess(crop[3], hip[3])

    def test_connected_blank_image_reports_cutout_failure(self):
        blank = Image.new("RGB", (256, 256), "white")
        spec = {"items": {"top": {"item_id": "blank", "slot": "top"}}}
        with self.assertRaisesRegex(ValueError, "top image has no detectable foreground"):
            self.node.compose(
                json.dumps(spec),
                768,
                1024,
                "white",
                False,
                top=image_tensor(blank),
            )

    def test_image_batch_is_rejected_instead_of_truncated(self):
        batch = torch.zeros((2, 128, 128, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "exactly one IMAGE"):
            self.node._image(batch)


if __name__ == "__main__":
    unittest.main()
