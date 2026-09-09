import json
import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import OutfitReferenceComposer, SLOTS, TEMPLATE  # noqa: E402


def image_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def axis_change(source, placement):
    source_aspect = source.width / source.height
    target_aspect = placement[2] / placement[3]
    return max(source_aspect / target_aspect, target_aspect / source_aspect)


class OutfitReferenceComposerTests(unittest.TestCase):
    def setUp(self):
        self.node = OutfitReferenceComposer()

    def test_node_has_socks_input_and_one_output(self):
        self.assertIn("socks", SLOTS)
        optional = list(self.node.INPUT_TYPES()["optional"])
        self.assertLess(optional.index("socks"), optional.index("shoes"))
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

    def test_top_resize_preserves_source_aspect(self):
        for cutout, layout in (
            (
                Image.new("RGBA", (680, 480), (80, 20, 20, 255)),
                {"garment_type": "jersey", "fit": "loose", "length": "crop"},
            ),
            (
                Image.new("RGBA", (530, 650), (30, 80, 130, 255)),
                {"garment_type": "shirt", "fit": "loose", "length": "hip"},
            ),
        ):
            placement, _, _ = self.node._top_placement(cutout, layout, 768, 1024)
            self.assertLessEqual(axis_change(cutout, placement), 1.02)

    def test_bottom_resize_preserves_source_aspect(self):
        cutout = Image.new("RGBA", (300, 600), (80, 80, 80, 255))
        placement, _, _ = self.node._bottom_placement(
            cutout,
            {"garment_type": "pants", "fit": "oversized", "waist": "low", "length": "floor", "silhouette": "baggy"},
            768,
            1024,
        )
        self.assertLessEqual(axis_change(cutout, placement), 1.02)

    def test_product_shot_aspect_is_not_normalized_away(self):
        narrow = Image.new("RGBA", (360, 400), (80, 80, 80, 255))
        wide = Image.new("RGBA", (620, 420), (80, 80, 80, 255))
        layout = {"garment_type": "shorts", "fit": "loose", "waist": "mid", "length": "knee", "silhouette": "wide"}
        narrow_placement, _, _ = self.node._bottom_placement(narrow, layout, 768, 1024)
        wide_placement, _, _ = self.node._bottom_placement(wide, layout, 768, 1024)
        self.assertLessEqual(axis_change(narrow, narrow_placement), 1.02)
        self.assertLessEqual(axis_change(wide, wide_placement), 1.02)
        self.assertNotEqual(narrow_placement[2:], wide_placement[2:])

    def test_extreme_aspect_is_contained_without_distortion(self):
        extreme = Image.new("RGBA", (1000, 100), (80, 80, 80, 255))
        placement, hard, _ = self.node._top_placement(
            extreme,
            {"garment_type": "shirt", "fit": "regular", "length": "hip", "max_axis_distortion": 4},
            768,
            1024,
        )
        self.assertLessEqual(axis_change(extreme, placement), 1.02)
        self.assertLessEqual(placement[2], hard[2] - hard[0])

    def test_packed_main_slots_do_not_overlap(self):
        top = Image.new("RGBA", (400, 320), (80, 80, 80, 255))
        bottom = Image.new("RGBA", (260, 500), (80, 80, 80, 255))
        shoes = Image.new("RGBA", (300, 350), (80, 80, 80, 255))
        layouts = {
            "top": {"garment_type": "shirt", "fit": "oversized", "length": "upper_thigh"},
            "bottom": {"garment_type": "pants", "fit": "oversized", "length": "floor", "silhouette": "baggy"},
            "shoes": {"size": "large"},
        }
        placements = {
            "top": self.node._top_placement(top, layouts["top"], 768, 1024),
            "bottom": self.node._bottom_placement(bottom, layouts["bottom"], 768, 1024),
            "shoes": self.node._box_placement("shoes", shoes, layouts["shoes"], 768, 1024),
        }
        packed = self.node._pack_main(placements, layouts, 768, 1024)
        previous_bottom = None
        for slot in ("top", "bottom", "shoes"):
            placement, hard, _ = packed[slot]
            if previous_bottom is not None:
                self.assertGreaterEqual(placement[1] - previous_bottom, 8)
            self.assertGreaterEqual(placement[0], hard[0])
            self.assertLessEqual(placement[0] + placement[2], hard[2])
            previous_bottom = placement[1] + placement[3]
        self.assertLessEqual(previous_bottom, round(TEMPLATE["main_bottom_y"] * 1024))

    def test_bottom_waist_levels_are_anchored_above_shared_hip_axis(self):
        bottom = Image.new("RGBA", (300, 500), (80, 80, 80, 255))
        positions = {}
        for waist in ("high", "mid", "low"):
            placement, _, _ = self.node._bottom_placement(
                bottom,
                {
                    "garment_type": "pants",
                    "fit": "regular",
                    "waist": waist,
                    "length": "floor",
                    "silhouette": "straight",
                },
                768,
                1024,
            )
            positions[waist] = placement[1]

        self.assertLess(positions["high"], positions["mid"])
        self.assertLess(positions["mid"], positions["low"])
        hip_y = round(TEMPLATE["landmarks"]["hip_y"] * 1024)
        for waist, y in positions.items():
            expected = round(
                (
                    TEMPLATE["landmarks"]["hip_y"]
                    - TEMPLATE["landmarks"]["waist_offset_from_hip"][waist]
                )
                * 1024
            )
            self.assertEqual(y, expected)
            self.assertLess(y, hip_y)

    def test_isolated_shoes_keep_fixed_foot_anchor(self):
        shoes = Image.new("RGBA", (300, 350), (80, 80, 80, 255))
        layouts = {"shoes": {"size": "large"}}
        original = self.node._box_placement("shoes", shoes, layouts["shoes"], 768, 1024)
        packed = self.node._pack_main({"shoes": original}, layouts, 768, 1024)
        self.assertEqual(packed["shoes"][0], original[0])

    def test_missing_bottom_does_not_pull_shoes_below_top(self):
        top = Image.new("RGBA", (400, 360), (80, 80, 80, 255))
        shoes = Image.new("RGBA", (300, 350), (80, 80, 80, 255))
        layouts = {
            "top": {"garment_type": "shirt", "fit": "regular", "length": "waist"},
            "shoes": {"size": "large"},
        }
        shoe_original = self.node._box_placement("shoes", shoes, layouts["shoes"], 768, 1024)
        placements = {
            "top": self.node._top_placement(top, layouts["top"], 768, 1024),
            "shoes": shoe_original,
        }
        packed = self.node._pack_main(placements, layouts, 768, 1024)
        self.assertEqual(packed["shoes"][0], shoe_original[0])
        top_box = packed["top"][0]
        shoe_box = packed["shoes"][0]
        self.assertGreater(shoe_box[1], top_box[1] + top_box[3])

    def test_every_main_slot_subset_stays_inside_and_nonoverlapping(self):
        cutouts = {
            "top": Image.new("RGBA", (400, 360), (80, 80, 80, 255)),
            "bottom": Image.new("RGBA", (340, 360), (80, 80, 80, 255)),
            "socks": Image.new("RGBA", (220, 300), (230, 230, 230, 255)),
            "shoes": Image.new("RGBA", (300, 350), (80, 80, 80, 255)),
        }
        layouts = {
            "top": {"garment_type": "shirt", "fit": "regular", "length": "waist"},
            "bottom": {
                "garment_type": "shorts",
                "fit": "regular",
                "waist": "mid",
                "length": "knee",
                "silhouette": "straight",
            },
            "socks": {"length": "mid_calf"},
            "shoes": {"size": "large"},
        }
        all_placements = {
            "top": self.node._top_placement(cutouts["top"], layouts["top"], 768, 1024),
            "bottom": self.node._bottom_placement(
                cutouts["bottom"], layouts["bottom"], 768, 1024
            ),
            "socks": self.node._socks_placement(
                cutouts["socks"], layouts["socks"], layouts["bottom"], 768, 1024, True
            ),
            "shoes": self.node._box_placement(
                "shoes", cutouts["shoes"], layouts["shoes"], 768, 1024
            ),
        }

        slots = tuple(all_placements)
        for count in range(1, len(slots) + 1):
            for subset in combinations(slots, count):
                with self.subTest(subset=subset):
                    packed = self.node._pack_main(
                        {slot: all_placements[slot] for slot in subset}, layouts, 768, 1024
                    )
                    boxes = [packed[slot][0] for slot in subset]
                    for box in boxes:
                        self.assertGreaterEqual(box[0], 0)
                        self.assertGreaterEqual(box[1], 0)
                        self.assertLessEqual(box[0] + box[2], 768)
                        self.assertLessEqual(box[1] + box[3], 1024)
                    for first, second in combinations(boxes, 2):
                        horizontal_overlap = min(first[0] + first[2], second[0] + second[2]) - max(
                            first[0], second[0]
                        )
                        vertical_overlap = min(first[1] + first[3], second[1] + second[3]) - max(
                            first[1], second[1]
                        )
                        self.assertFalse(horizontal_overlap > 0 and vertical_overlap > 0)

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

    def test_short_bottom_and_socks_are_packed_without_overlap(self):
        bottom = Image.new("RGBA", (360, 250), (80, 80, 80, 255))
        socks = Image.new("RGBA", (220, 300), (230, 230, 230, 255))
        layouts = {
            "bottom": {"garment_type": "shorts", "fit": "loose", "waist": "low", "length": "knee", "silhouette": "wide"},
            "socks": {"length": "mid_calf"},
        }
        placements = {
            "bottom": self.node._bottom_placement(bottom, layouts["bottom"], 768, 1024),
            "socks": self.node._socks_placement(
                socks, layouts["socks"], layouts["bottom"], 768, 1024, True
            ),
        }
        packed = self.node._pack_main(placements, layouts, 768, 1024)
        bottom_box = packed["bottom"][0]
        socks_box = packed["socks"][0]
        self.assertGreaterEqual(socks_box[1] - (bottom_box[1] + bottom_box[3]), 8)

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

    def test_fit_and_length_change_uniform_top_scale_monotonically(self):
        cutout = Image.new("RGBA", (400, 420), (80, 80, 80, 255))
        widths = []
        for fit in ("slim", "regular", "loose", "oversized"):
            placement, _, _ = self.node._top_placement(
                cutout,
                {"garment_type": "shirt", "fit": fit, "length": "waist"},
                768,
                1024,
            )
            widths.append(placement[2])
            self.assertLessEqual(axis_change(cutout, placement), 1.02)
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(len(widths), len(set(widths)))

        heights = []
        for length in ("crop", "waist", "hip", "upper_thigh"):
            placement, _, _ = self.node._top_placement(
                cutout,
                {"garment_type": "shirt", "fit": "regular", "length": length},
                768,
                1024,
            )
            heights.append(placement[3])
        self.assertEqual(heights, sorted(heights))
        self.assertEqual(len(heights), len(set(heights)))

    def test_bottom_fit_and_length_change_uniform_scale_monotonically(self):
        cutout = Image.new("RGBA", (300, 500), (80, 80, 80, 255))
        widths = []
        for fit in ("slim", "regular", "loose", "oversized"):
            placement, _, _ = self.node._bottom_placement(
                cutout,
                {"garment_type": "pants", "fit": fit, "length": "calf", "silhouette": "straight"},
                768,
                1024,
            )
            widths.append(placement[2])
            self.assertLessEqual(axis_change(cutout, placement), 1.02)
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(len(widths), len(set(widths)))

        heights = []
        for length in ("thigh", "knee", "calf", "ankle", "floor"):
            placement, _, _ = self.node._bottom_placement(
                cutout,
                {"garment_type": "pants", "fit": "regular", "length": length, "silhouette": "straight"},
                768,
                1024,
            )
            heights.append(placement[3])
        self.assertEqual(heights, sorted(heights))

    def test_accessories_use_stylised_character_scale(self):
        glasses = Image.new("RGBA", (430, 190), (80, 80, 80, 255))
        glasses_box, _, _ = self.node._box_placement(
            "glasses", glasses, {"size": "small"}, 768, 1024
        )
        self.assertGreaterEqual(glasses_box[2], round(768 * 0.135))
        self.assertLessEqual(axis_change(glasses, glasses_box), 1.02)

        hat = Image.new("RGBA", (300, 190), (80, 80, 80, 255))
        hat_box, _, _ = self.node._box_placement(
            "hat", hat, {"size": "medium"}, 768, 1024
        )
        self.assertGreaterEqual(hat_box[3], round(1024 * 0.085))
        self.assertLessEqual(axis_change(hat, hat_box), 1.02)

        shoes = Image.new("RGBA", (300, 350), (80, 80, 80, 255))
        shoes_box, _, _ = self.node._box_placement(
            "shoes", shoes, {"size": "large"}, 768, 1024
        )
        self.assertGreaterEqual(shoes_box[3], round(1024 * 0.13))
        self.assertLessEqual(axis_change(shoes, shoes_box), 1.02)

    def test_bracelet_defaults_left_and_normalises_visible_height(self):
        bead_strand = Image.new("RGBA", (260, 170), (80, 80, 80, 255))
        metal_bangles = Image.new("RGBA", (400, 150), (80, 80, 80, 255))
        bead_box, _, bead_label = self.node._box_placement(
            "bracelet", bead_strand, {"size": "medium"}, 768, 1024
        )
        metal_box, _, metal_label = self.node._box_placement(
            "bracelet", metal_bangles, {"size": "medium"}, 768, 1024
        )
        self.assertIn("left", bead_label)
        self.assertIn("left", metal_label)
        self.assertLess(bead_box[0] + bead_box[2], round(768 * 0.23) + 1)
        self.assertLess(metal_box[0] + metal_box[2], round(768 * 0.23) + 1)
        self.assertLessEqual(abs(bead_box[3] - metal_box[3]), 3)
        for box in (bead_box, metal_box):
            centre_y = box[1] + box[3] / 2
            self.assertAlmostEqual(centre_y, 1024 * 0.555, delta=1.5)
            self.assertLessEqual(axis_change(bead_strand if box == bead_box else metal_bangles, box), 1.02)

    def test_bracelet_can_be_explicitly_placed_on_right(self):
        bracelet = Image.new("RGBA", (280, 150), (80, 80, 80, 255))
        placement, _, label = self.node._box_placement(
            "bracelet",
            bracelet,
            {"size": "medium", "placement": "right_wrist"},
            768,
            1024,
        )
        self.assertGreaterEqual(placement[0], round(768 * 0.775))
        self.assertIn("right", label)

    def test_bag_uses_handle_top_as_hand_anchor(self):
        small_bag = Image.new("RGBA", (240, 300), (80, 80, 80, 255))
        large_bag = Image.new("RGBA", (300, 360), (80, 80, 80, 255))
        small_box, _, _ = self.node._box_placement(
            "bag", small_bag, {"size": "small"}, 768, 1024
        )
        large_box, _, _ = self.node._box_placement(
            "bag", large_bag, {"size": "large"}, 768, 1024
        )
        expected_y = round(TEMPLATE["landmarks"]["bag_handle_y"] * 1024)
        self.assertEqual(small_box[1], expected_y)
        self.assertEqual(large_box[1], expected_y)

    def test_template_has_explicit_hip_axis(self):
        self.assertAlmostEqual(TEMPLATE["landmarks"]["hip_y"], 0.49)

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
