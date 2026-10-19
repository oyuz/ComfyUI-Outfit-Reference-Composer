import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    OutfitReferenceComposerV2,
    OutfitReferenceComposerV3,
    OutfitReferenceComposerV3Top,
    SLOTS,
    V2_DRAW_ORDER,
    V2_FIXED_SLOTS,
    V3_BOTTOM_SHOEHALF_COVER_FRACTION,
    V3_BOTTOM_SHOEHALF_SEAM_OVERLAP_FRACTION,
    V3_TOP_COVERWAIST_HIP_FRACTION,
)


def rgba_tensor(size, color):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(
        (4, 4, size[0] - 5, size[1] - 5),
        fill=color,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class OutfitReferenceComposerV3Tests(unittest.TestCase):
    def setUp(self):
        self.v2 = OutfitReferenceComposerV2()
        self.v3 = OutfitReferenceComposerV3()

    def images(self):
        return {
            "top": rgba_tensor((400, 600), (230, 20, 20, 255)),
            "bottom": rgba_tensor((300, 700), (20, 180, 50, 255)),
            "shoes": rgba_tensor((400, 200), (20, 60, 230, 255)),
        }

    def render_v2(self, **images):
        return self.v2.compose_fixed(
            1000, 1000, "white", 24, 0, False, **images
        )[0]

    def render_v3(self, tags="{}", **images):
        return self.v3.compose_v3(
            1000,
            1000,
            "white",
            24,
            0,
            False,
            layout_tags_json=tags,
            **images,
        )[0]

    def test_v2_v3_and_v3_top_are_registered_as_distinct_nodes(self):
        self.assertIs(
            NODE_CLASS_MAPPINGS["OutfitReferenceComposerV2"],
            OutfitReferenceComposerV2,
        )
        self.assertIs(
            NODE_CLASS_MAPPINGS["OutfitReferenceComposerV3"],
            OutfitReferenceComposerV3,
        )
        self.assertIs(
            NODE_CLASS_MAPPINGS["OutfitReferenceComposerV3Top"],
            OutfitReferenceComposerV3Top,
        )
        self.assertEqual(len({OutfitReferenceComposerV2, OutfitReferenceComposerV3, OutfitReferenceComposerV3Top}), 3)

    def test_v3_appends_one_metadata_input_after_the_v2_image_inputs(self):
        optional = list(self.v3.INPUT_TYPES()["optional"])
        self.assertEqual(optional[:-1], list(SLOTS))
        self.assertEqual(optional[-1], "layout_tags_json")
        self.assertTrue(
            self.v3.INPUT_TYPES()["optional"]["layout_tags_json"][1]["forceInput"]
        )

    def test_empty_tags_are_pixel_identical_to_v2(self):
        images = self.images()
        self.assertTrue(torch.equal(self.render_v2(**images), self.render_v3(**images)))

    def test_tag_without_required_counterpart_falls_back_to_v2(self):
        all_images = self.images()
        for images, tagged in (
            (
                {"top": all_images["top"]},
                json.dumps({"top": "top_coverwaist"}),
            ),
            (
                {"bottom": all_images["bottom"]},
                json.dumps({"bottom": "bottom_shoehalf"}),
            ),
        ):
            with self.subTest(tagged=tagged):
                self.assertTrue(
                    torch.equal(
                        self.render_v2(**images),
                        self.render_v3(tagged, **images),
                    )
                )

    def test_tags_are_strict_and_actionable(self):
        for payload, message in (
            ("not-json", "not valid JSON"),
            ("[]", "must be a JSON object"),
            ('{"shoes":"bottom_shoehalf"}', "only accepts top and bottom"),
            ('{"top":"bottom_shoehalf"}', "must be 'top_coverwaist'"),
            ('{"bottom":"top_coverwaist"}', "must be 'bottom_shoehalf'"),
            ('{"top":"coverwaist"}', "must be 'top_coverwaist'"),
            ('{"bottom":"shoehalf"}', "must be 'bottom_shoehalf'"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, message
            ):
                self.render_v3(payload, **self.images())

    def test_relative_placements_target_hip_and_shoe_thirds(self):
        images = self.images()
        cutouts = {
            slot: self.v3._cutout(
                self.v3._image(tensor), slot, {"background_threshold": 24}
            )
            for slot, tensor in images.items()
        }
        base = {
            slot: self.v3._fit_fixed_slot(cutout, slot, 1536, 2048, 10)[0]
            for slot, cutout in cutouts.items()
        }
        placements, hard_boxes = self.v3._v3_relative_placements(
            cutouts, 1536, 2048, 10, True, True
        )
        top_placement = placements["top"]
        bottom_placement = placements["bottom"]
        shoe_placement = placements["shoes"]

        top_bottom = top_placement[1] + top_placement[3]
        shoe_top = shoe_placement[1]
        shoe_height = shoe_placement[3]
        expected_hip = bottom_placement[1] + round(
            bottom_placement[3] * V3_TOP_COVERWAIST_HIP_FRACTION
        )
        expected_shoe_boundary = shoe_top + round(
            shoe_height * V3_BOTTOM_SHOEHALF_COVER_FRACTION
        )
        expected_visible_hem = expected_shoe_boundary + max(
            1,
            round(shoe_height * V3_BOTTOM_SHOEHALF_SEAM_OVERLAP_FRACTION),
        )
        self.assertEqual(top_bottom, expected_hip)
        visible_bbox = self.v3._visible_alpha_bbox(cutouts["bottom"])
        mapped_visible_hem = bottom_placement[1] + (
            visible_bbox[3] * bottom_placement[3] / cutouts["bottom"].height
        )
        self.assertGreaterEqual(mapped_visible_hem, expected_visible_hem)
        self.assertLess(mapped_visible_hem, expected_visible_hem + 1.1)

        # Tagged garments retain V2's upper edge and their semantic centre
        # axis while scaling uniformly to the neighbouring item.
        self.assertEqual(top_placement[1], base["top"][1])
        self.assertEqual(bottom_placement[1], base["bottom"][1])
        outfit_axis = 0.46 * 1536
        self.assertAlmostEqual(
            top_placement[0] + top_placement[2] / 2,
            outfit_axis,
            delta=1,
        )
        waist_center = self.v3._bottom_waist_center(cutouts["bottom"])
        self.assertAlmostEqual(
            bottom_placement[0]
            + waist_center * bottom_placement[2] / cutouts["bottom"].width,
            outfit_axis,
            delta=1,
        )
        for slot in ("top", "bottom"):
            source_aspect = cutouts[slot].width / cutouts[slot].height
            placed_aspect = placements[slot][2] / placements[slot][3]
            self.assertAlmostEqual(placed_aspect, source_aspect, delta=0.005)
            actual_box = self.v3._placement_box(placements[slot])
            hard = hard_boxes[slot]
            self.assertLessEqual(hard[0], actual_box[0])
            self.assertLessEqual(hard[1], actual_box[1])
            self.assertGreaterEqual(hard[2], actual_box[2])
            self.assertGreaterEqual(hard[3], actual_box[3])

    def test_shoe_layer_is_cropped_to_its_lower_third_at_same_scale(self):
        shoe = self.v3._cutout(
            self.v3._image(self.images()["shoes"]),
            "shoes",
            {"background_threshold": 24},
        )
        placement, _ = self.v3._fit_fixed_slot(shoe, "shoes", 1536, 2048, 10)
        toe, toe_placement = self.v3._shoe_toe_third(shoe, placement)
        boundary = placement[1] + round(
            placement[3] * V3_BOTTOM_SHOEHALF_COVER_FRACTION
        )
        self.assertEqual(toe_placement[1], boundary)
        self.assertEqual(toe_placement[0], placement[0])
        self.assertEqual(toe_placement[2], placement[2])
        self.assertEqual(
            toe_placement[1] + toe_placement[3],
            placement[1] + placement[3],
        )
        source_aspect = toe.width / toe.height
        placed_aspect = toe_placement[2] / toe_placement[3]
        self.assertAlmostEqual(placed_aspect, source_aspect, delta=0.08)

    def test_shoe_pixels_cannot_leak_above_the_toe_third(self):
        images = self.images()
        cutouts = {
            slot: self.v3._cutout(
                self.v3._image(tensor), slot, {"background_threshold": 24}
            )
            for slot, tensor in images.items()
        }
        placements, _ = self.v3._v3_relative_placements(
            cutouts, 1000, 1000, 0, True, True
        )
        _toe, toe_placement = self.v3._shoe_toe_third(
            cutouts["shoes"], placements["shoes"]
        )
        pixels = self.render_v3(
            json.dumps(
                {"top": "top_coverwaist", "bottom": "bottom_shoehalf"}
            ),
            **images,
        )[0].numpy()
        blue = (
            (pixels[..., 2] > 0.65)
            & (pixels[..., 0] < 0.25)
            & (pixels[..., 1] < 0.45)
        )
        visible_rows = np.flatnonzero(np.any(blue, axis=1))
        self.assertGreater(visible_rows.size, 0)
        self.assertGreaterEqual(int(visible_rows[0]), toe_placement[1])

    def test_visible_bottom_hem_fills_transparent_seam(self):
        output = self.v3.compose_v3(
            1000,
            1000,
            "off_white",
            24,
            0,
            False,
            layout_tags_json=json.dumps({"bottom": "bottom_shoehalf"}),
            **self.images(),
        )[0][0].numpy()
        images = self.images()
        cutouts = {
            slot: self.v3._cutout(
                self.v3._image(tensor), slot, {"background_threshold": 24}
            )
            for slot, tensor in images.items()
        }
        placements, _ = self.v3._v3_relative_placements(
            cutouts, 1000, 1000, 0, False, True
        )
        nominal_boundary = placements["shoes"][1] + round(
            placements["shoes"][3] * V3_BOTTOM_SHOEHALF_COVER_FRACTION
        )
        axis_x = round(0.46 * 1000)

        # The synthetic bottom cutout retains transparent padding.  V3 maps
        # the visible alpha hem (rather than the padded rectangle) to the
        # nominal boundary, so the row immediately above it is garment rather
        # than the off-white canvas.
        seam_pixel = output[nominal_boundary - 1, axis_x]
        canvas = np.array([248, 248, 245], dtype=np.float32) / 255.0
        self.assertGreater(float(np.max(np.abs(seam_pixel - canvas))), 0.20)
        self.assertGreater(seam_pixel[1], 0.60)

    def test_draw_order_puts_the_covering_item_on_top(self):
        self.assertEqual(
            self.v3._v3_draw_order(True, True)[:4],
            ("socks", "shoes", "bottom", "top"),
        )
        self.assertGreater(
            self.v3._v3_draw_order(True, False).index("top"),
            self.v3._v3_draw_order(True, False).index("bottom"),
        )
        self.assertGreater(
            self.v3._v3_draw_order(False, True).index("bottom"),
            self.v3._v3_draw_order(False, True).index("shoes"),
        )

    def test_both_tags_visibly_control_the_two_overlap_regions(self):
        output = self.render_v3(
            json.dumps(
                {"top": "top_coverwaist", "bottom": "bottom_shoehalf"}
            ),
            **self.images(),
        )[0].numpy()

        # The top is drawn after the bottom at their waist overlap.
        waist_pixel = output[500, 460]
        self.assertGreater(waist_pixel[0], 0.8)
        self.assertLess(waist_pixel[1], 0.2)

        # The bottom is drawn after the shoes at their lower overlap.
        shoe_pixel = output[850, 460]
        self.assertGreater(shoe_pixel[1], 0.6)
        self.assertLess(shoe_pixel[2], 0.3)

    def test_v3_render_never_mutates_v2_constants(self):
        boxes = copy.deepcopy(V2_FIXED_SLOTS)
        order = tuple(V2_DRAW_ORDER)
        self.render_v3(
            '{"top":"top_coverwaist","bottom":"bottom_shoehalf"}',
            **self.images(),
        )
        self.assertEqual(V2_FIXED_SLOTS, boxes)
        self.assertEqual(V2_DRAW_ORDER, order)


if __name__ == "__main__":
    unittest.main()
