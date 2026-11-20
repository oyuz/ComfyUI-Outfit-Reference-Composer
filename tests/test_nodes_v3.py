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
    V3_FIXED_SLOTS,
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

    def render_v3(self, tags="{}", guides=False, **images):
        return self.v3.compose_v3(
            1000,
            1000,
            "white",
            24,
            0,
            guides,
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
        self.assertNotIn("socks", optional)
        self.assertEqual(optional[-1], "layout_tags_json")
        self.assertTrue(
            self.v3.INPUT_TYPES()["optional"]["layout_tags_json"][1]["forceInput"]
        )

    def test_removed_socks_input_is_rejected_instead_of_ignored(self):
        with self.assertRaisesRegex(
            ValueError, "unsupported outfit image inputs: socks"
        ):
            self.render_v3(socks=rgba_tensor((160, 220), (80, 90, 100, 255)))

    def test_complete_untagged_set_uses_v3_side_necklace_and_full_canvas(self):
        images = {
            slot: rgba_tensor(
                (180 + index * 9, 160 + index * 13),
                (30 + index * 12, 80, 170, 255),
            )
            for index, slot in enumerate(SLOTS)
        }
        v2 = self.render_v2(**images)
        v3 = self.render_v3(**images)
        self.assertEqual(tuple(v3.shape), (1, 1000, 1000, 3))
        self.assertFalse(torch.equal(v2, v3))

        cutouts = {
            slot: self.v3._cutout(
                self.v3._image(tensor), slot, {"background_threshold": 24}
            )
            for slot, tensor in images.items()
        }
        boxes = self.v3._dynamic_boxes(cutouts, 1000, 1000)
        placements, _ = self.v3._v3_relative_placements(
            cutouts, 1000, 1000, 0, False, False, boxes
        )
        top_box = self.v3._placement_box(placements["top"])
        necklace_box = self.v3._placement_box(placements["necklace"])
        self.assertGreaterEqual(necklace_box[0], top_box[2])

    def test_incomplete_untagged_set_compacts_and_crops(self):
        images = self.images()
        v2 = self.render_v2(**images)
        v3 = self.render_v3(**images)
        self.assertNotEqual(tuple(v2.shape), tuple(v3.shape))
        self.assertLess(v3.shape[1], v2.shape[1])
        self.assertLess(v3.shape[2], v2.shape[2])
        self.assertEqual(v3.shape[1] % 64, 0)
        self.assertEqual(v3.shape[2] % 64, 0)
        guided = self.render_v3(guides=True, **images)
        self.assertEqual(tuple(guided.shape), (1, 1000, 1000, 3))

    def test_tag_without_required_counterpart_uses_same_dynamic_render_as_no_tag(self):
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
                self.assertTrue(torch.equal(
                    self.render_v3("{}", **images),
                    self.render_v3(tagged, **images),
                ))

    def test_full_semantic_rows_keep_original_hard_boxes(self):
        cutouts = {
            "hat": Image.new("RGBA", (80, 60), (30, 40, 50, 255)),
            "glasses": Image.new("RGBA", (100, 40), (30, 40, 50, 255)),
            "top": Image.new("RGBA", (160, 240), (30, 40, 50, 255)),
            "bottom": Image.new("RGBA", (140, 260), (30, 40, 50, 255)),
            "shoes": Image.new("RGBA", (160, 80), (30, 40, 50, 255)),
        }
        boxes = self.v3._dynamic_boxes(cutouts, 1536, 2048)
        for name in ("hat", "glasses", "top", "bottom", "shoes"):
            with self.subTest(name=name):
                self.assertEqual(
                    boxes[name],
                    self.v3._normal_box_to_pixels(V2_FIXED_SLOTS[name], 1536, 2048),
                )
        self.assertEqual(
            boxes["necklace"],
            self.v3._normal_box_to_pixels(V3_FIXED_SLOTS["necklace"], 1536, 2048),
        )

    def test_necklace_only_stays_in_side_slot_and_crops_safely(self):
        necklace = rgba_tensor((240, 180), (20, 70, 230, 255))
        cutout = self.v3._cutout(
            self.v3._image(necklace), "necklace", {"background_threshold": 24}
        )
        boxes = self.v3._dynamic_boxes({"necklace": cutout}, 1000, 1000)
        placement, _ = self.v3._fit_dynamic_slot(
            cutout, "necklace", boxes, 0
        )
        self.assertGreaterEqual(placement[0], 760)
        output = self.render_v3(necklace=necklace)
        self.assertLess(output.shape[1], 1000)
        self.assertLess(output.shape[2], 1000)
        self.assertEqual(output.shape[1] % 64, 0)
        self.assertEqual(output.shape[2] % 64, 0)

    def test_empty_face_row_is_reused_by_hat(self):
        hat = {"hat": Image.new("RGBA", (120, 80), (30, 40, 50, 255))}
        hat_only = self.v3._dynamic_boxes(hat, 1000, 1000)["hat"]
        with_face = self.v3._dynamic_boxes(
            {**hat, "glasses": Image.new("RGBA", (120, 50), (30, 40, 50, 255))},
            1000,
            1000,
        )["hat"]
        self.assertEqual(hat_only[1], with_face[1])
        self.assertGreater(hat_only[3], with_face[3])

    def test_rgba_alpha_is_the_only_cutout_mask(self):
        image = Image.new("RGBA", (80, 80), (12, 34, 56, 0))
        ImageDraw.Draw(image).rectangle((20, 18, 59, 61), fill=(220, 30, 30, 255))
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        low = self.v3._cutout(
            self.v3._image(tensor), "top", {"background_threshold": 4}
        )
        high = self.v3._cutout(
            self.v3._image(tensor), "top", {"background_threshold": 100}
        )
        self.assertTrue(np.array_equal(np.asarray(low), np.asarray(high)))
        cutout = low
        self.assertLess(cutout.width, image.width)
        self.assertLess(cutout.height, image.height)

    def test_v2_v3_and_v3_top_reject_rgb_input_without_alpha(self):
        rgb = torch.zeros((1, 80, 80, 3), dtype=torch.float32)
        renderers = (
            lambda: self.v2.compose_fixed(
                768, 1024, "white", 24, 8, False, top=rgb
            ),
            lambda: self.v3.compose_v3(
                768, 1024, "white", 24, 8, False, "{}", top=rgb
            ),
            lambda: OutfitReferenceComposerV3Top().compose_top(
                768, 1024, "white", 24, 8, False, top=rgb
            ),
        )
        for render in renderers:
            with self.subTest(render=render), self.assertRaisesRegex(
                ValueError, "four-channel RGBA IMAGE.*Preserve Alpha"
            ):
                render()

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

    def test_relative_placements_target_hip_and_shoe_half(self):
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
        self.assertEqual(top_bottom, expected_hip)
        visible_bbox = self.v3._visible_alpha_bbox(cutouts["bottom"])
        mapped_visible_hem = bottom_placement[1] + (
            visible_bbox[3] * bottom_placement[3] / cutouts["bottom"].height
        )
        self.assertGreaterEqual(mapped_visible_hem, expected_shoe_boundary)
        self.assertLess(mapped_visible_hem, expected_shoe_boundary + 1.1)

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

    def test_overlap_tags_apply_after_empty_row_reallocation(self):
        images = self.images()
        cutouts = {
            slot: self.v3._cutout(
                self.v3._image(tensor), slot, {"background_threshold": 24}
            )
            for slot, tensor in images.items()
        }

        upper_pair = {slot: cutouts[slot] for slot in ("top", "bottom")}
        upper_boxes = self.v3._dynamic_boxes(upper_pair, 1000, 1000)
        upper_placements, _ = self.v3._v3_relative_placements(
            upper_pair,
            1000,
            1000,
            0,
            True,
            False,
            upper_boxes,
        )
        self.assertEqual(
            upper_placements["top"][1] + upper_placements["top"][3],
            upper_placements["bottom"][1]
            + round(
                upper_placements["bottom"][3]
                * V3_TOP_COVERWAIST_HIP_FRACTION
            ),
        )

        lower_pair = {slot: cutouts[slot] for slot in ("bottom", "shoes")}
        lower_boxes = self.v3._dynamic_boxes(lower_pair, 1000, 1000)
        lower_placements, _ = self.v3._v3_relative_placements(
            lower_pair,
            1000,
            1000,
            0,
            False,
            True,
            lower_boxes,
        )
        shoe_top = lower_placements["shoes"][1]
        shoe_height = lower_placements["shoes"][3]
        visible_bbox = self.v3._visible_alpha_bbox(lower_pair["bottom"])
        mapped_hem = lower_placements["bottom"][1] + (
            visible_bbox[3]
            * lower_placements["bottom"][3]
            / lower_pair["bottom"].height
        )
        expected_boundary = shoe_top + round(
            shoe_height * V3_BOTTOM_SHOEHALF_COVER_FRACTION
        )
        self.assertGreaterEqual(mapped_hem, expected_boundary)
        self.assertLess(mapped_hem, expected_boundary + 1.1)

    def test_full_shoe_is_drawn_then_pants_cover_its_centre(self):
        images = {
            "bottom": rgba_tensor((120, 700), (20, 180, 50, 255)),
            "shoes": rgba_tensor((400, 300), (20, 60, 230, 255)),
        }
        cutouts = {
            slot: self.v3._cutout(
                self.v3._image(tensor), slot, {"background_threshold": 24}
            )
            for slot, tensor in images.items()
        }
        boxes = self.v3._dynamic_boxes(cutouts, 1000, 1000)
        placements, _ = self.v3._v3_relative_placements(
            cutouts, 1000, 1000, 0, False, True, boxes
        )
        pixels = self.render_v3(
            json.dumps({"bottom": "bottom_shoehalf"}),
            guides=True,
            **images,
        )[0].numpy()

        shoe = placements["shoes"]
        bottom = placements["bottom"]
        self.assertLess(shoe[0], bottom[0])
        self.assertGreater(shoe[0] + shoe[2], bottom[0] + bottom[2])
        overlap_y = shoe[1] + max(2, shoe[3] // 6)
        self.assertLess(
            overlap_y,
            shoe[1] + round(shoe[3] * V3_BOTTOM_SHOEHALF_COVER_FRACTION),
        )

        row = pixels[overlap_y]
        blue = (
            (row[..., 2] > 0.65)
            & (row[..., 0] < 0.25)
            & (row[..., 1] < 0.45)
        )
        green = (
            (row[..., 1] > 0.60)
            & (row[..., 0] < 0.25)
            & (row[..., 2] < 0.35)
        )
        # Green in the centre proves that the bottom is drawn over the shoe.
        # Blue at the same upper-shoe row, outside the narrow trouser alpha,
        # proves that the original full shoe was not source-cropped.
        self.assertTrue(np.any(green[bottom[0] : bottom[0] + bottom[2]]))
        self.assertTrue(np.any(blue[shoe[0] : bottom[0]]))
        self.assertTrue(
            np.any(blue[bottom[0] + bottom[2] : shoe[0] + shoe[2]])
        )

    def test_visible_bottom_hem_fills_transparent_seam(self):
        output = self.v3.compose_v3(
            1000,
            1000,
            "off_white",
            24,
            0,
            # Keep the requested coordinate system while inspecting the seam.
            # Guide strokes do not touch the sampled interior pixel below.
            True,
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
        # nominal half-height overlap boundary, so the row immediately above it
        # is the garment while the complete shoe remains underneath.
        seam_pixel = output[nominal_boundary - 1, axis_x + 5]
        canvas = np.array([248, 248, 245], dtype=np.float32) / 255.0
        self.assertGreater(float(np.max(np.abs(seam_pixel - canvas))), 0.20)
        self.assertGreater(seam_pixel[1], 0.60)

    def test_draw_order_puts_the_covering_item_on_top(self):
        self.assertEqual(
            self.v3._v3_draw_order(True, True)[:3],
            ("shoes", "bottom", "top"),
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
            guides=True,
            **self.images(),
        )[0].numpy()

        # The top is drawn after the bottom at their waist overlap.
        waist_pixel = output[500, 470]
        self.assertGreater(waist_pixel[0], 0.8)
        self.assertLess(waist_pixel[1], 0.2)

        # The bottom is drawn after the shoes at their lower overlap.
        shoe_pixel = output[850, 470]
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
