import copy
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
    OutfitReferenceComposerV3Top,
    SLOTS,
    V2_DRAW_ORDER,
    V2_FIXED_SLOTS,
    V2_OUTFIT_CENTER_X,
    V3_TOP_CENTER_X,
    V3_TOP_DRAW_ORDER,
    V3_TOP_FIXED_SLOTS,
    V3_TOP_SLOTS,
)


def rgba_tensor(image):
    array = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def product_tensor(size=(320, 260), color=(60, 90, 160, 255)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).rounded_rectangle(
        (4, 4, size[0] - 5, size[1] - 5), radius=8, fill=color
    )
    return rgba_tensor(image)


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def axis_change(source, placement):
    source_aspect = source.width / source.height
    target_aspect = placement[2] / placement[3]
    return max(source_aspect / target_aspect, target_aspect / source_aspect)


class OutfitReferenceComposerV3TopTests(unittest.TestCase):
    def setUp(self):
        self.v2 = OutfitReferenceComposerV2()
        self.v3 = OutfitReferenceComposerV3Top()

    def test_v2_and_v3_top_are_registered_and_callable_together(self):
        self.assertIs(
            NODE_CLASS_MAPPINGS["OutfitReferenceComposerV2"],
            OutfitReferenceComposerV2,
        )
        self.assertIs(
            NODE_CLASS_MAPPINGS["OutfitReferenceComposerV3Top"],
            OutfitReferenceComposerV3Top,
        )
        self.assertIsNot(OutfitReferenceComposerV2, OutfitReferenceComposerV3Top)

        top = product_tensor()
        before = top.clone()
        v2_output = getattr(self.v2, self.v2.FUNCTION)(
            768, 1024, "off_white", 24, 10, False, top=top,
            bottom=product_tensor((220, 400), (80, 60, 30, 255)),
        )[0]
        v3_output = getattr(self.v3, self.v3.FUNCTION)(
            768, 1024, "off_white", 24, 10, False, top=top,
        )[0]
        self.assertEqual(tuple(v2_output.shape), (1, 1024, 768, 3))
        self.assertEqual(tuple(v3_output.shape), (1, 1024, 768, 3))
        self.assertTrue(torch.equal(top, before))

    def test_v2_public_contract_is_unchanged(self):
        expected_slots = (
            "top", "bottom", "socks", "shoes", "hat", "bag",
            "glasses", "necklace", "earrings", "bracelet",
        )
        expected_boxes = {
            "hat": (0.31, 0.005, 0.61, 0.101),
            "glasses": (0.32, 0.102, 0.60, 0.168),
            "earrings_left": (0.23, 0.103, 0.32, 0.158),
            "earrings_right": (0.60, 0.103, 0.69, 0.158),
            "necklace": (0.35, 0.170, 0.57, 0.260),
            "top": (0.16, 0.175, 0.76, 0.475),
            "bottom": (0.20, 0.480, 0.72, 0.815),
            "bracelet": (0.01, 0.525, 0.17, 0.645),
            "bag": (0.73, 0.505, 0.99, 0.755),
            "socks": (0.09, 0.820, 0.25, 0.995),
            "shoes": (0.265, 0.820, 0.655, 0.995),
        }
        expected_draw_order = (
            "top", "bottom", "socks", "shoes", "hat", "earrings",
            "glasses", "necklace", "bag", "bracelet",
        )
        self.assertEqual(SLOTS, expected_slots)
        self.assertEqual(V2_OUTFIT_CENTER_X, 0.46)
        self.assertEqual(V2_FIXED_SLOTS, expected_boxes)
        self.assertEqual(V2_DRAW_ORDER, expected_draw_order)
        self.assertEqual(list(self.v2.INPUT_TYPES()["optional"]), list(expected_slots))
        self.assertEqual(self.v2.FUNCTION, "compose_fixed")
        self.assertEqual(self.v2.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(self.v2.RETURN_NAMES, ("outfit_reference",))

    def test_v3_top_exposes_only_five_supported_inputs(self):
        expected = ("top", "hat", "glasses", "earrings", "necklace")
        self.assertEqual(V3_TOP_SLOTS, expected)
        self.assertEqual(list(self.v3.INPUT_TYPES()["optional"]), list(expected))
        self.assertNotIn("outfit_spec", self.v3.INPUT_TYPES()["required"])
        forbidden = {"bottom", "socks", "shoes", "bag", "bracelet"}
        self.assertTrue(forbidden.isdisjoint(self.v3.INPUT_TYPES()["optional"]))
        for slot in sorted(forbidden):
            with self.subTest(slot=slot), self.assertRaisesRegex(
                ValueError, "unsupported inputs"
            ):
                self.v3.compose_top(
                    768, 1024, "white", 24, 8, False,
                    **{slot: product_tensor()},
                )

    def test_v3_top_uses_freed_canvas_and_one_center_axis(self):
        for name, box in V3_TOP_FIXED_SLOTS.items():
            with self.subTest(name=name):
                self.assertTrue(all(0.0 <= value <= 1.0 for value in box))
                self.assertLess(box[0], box[2])
                self.assertLess(box[1], box[3])

        v3_top = V3_TOP_FIXED_SLOTS["top"]
        v2_top = V2_FIXED_SLOTS["top"]
        v3_area = (v3_top[2] - v3_top[0]) * (v3_top[3] - v3_top[1])
        v2_area = (v2_top[2] - v2_top[0]) * (v2_top[3] - v2_top[1])
        self.assertGreater(v3_area, 3.5 * v2_area)
        self.assertLessEqual(min(box[0] for box in V3_TOP_FIXED_SLOTS.values()), 0.04)
        self.assertGreaterEqual(max(box[2] for box in V3_TOP_FIXED_SLOTS.values()), 0.96)
        self.assertLessEqual(min(box[1] for box in V3_TOP_FIXED_SLOTS.values()), 0.005)
        self.assertGreaterEqual(max(box[3] for box in V3_TOP_FIXED_SLOTS.values()), 0.995)

        for slot in ("hat", "glasses", "necklace", "top"):
            box = V3_TOP_FIXED_SLOTS[slot]
            self.assertAlmostEqual((box[0] + box[2]) / 2, V3_TOP_CENTER_X)
        hat = V3_TOP_FIXED_SLOTS["hat"]
        necklace = V3_TOP_FIXED_SLOTS["necklace"]
        self.assertGreaterEqual(hat[2] - hat[0] + 1e-9, 0.64)
        self.assertGreaterEqual(hat[3] - hat[1] + 1e-9, 0.22)
        self.assertGreaterEqual(
            (hat[3] - hat[1]) / (v3_top[3] - v3_top[1]), 0.33
        )
        self.assertGreaterEqual(necklace[2] - necklace[0] + 1e-9, 0.44)
        self.assertGreaterEqual(necklace[3] - necklace[1] + 1e-9, 0.20)
        self.assertGreaterEqual(necklace[1], 0.35)
        self.assertAlmostEqual((necklace[1] + necklace[3]) / 2, 0.45)
        self.assertGreaterEqual(
            (necklace[1] + necklace[3]) / 2 - v3_top[1], 0.10
        )
        self.assertLessEqual(
            V3_TOP_FIXED_SLOTS["glasses"][1] - V3_TOP_FIXED_SLOTS["hat"][3],
            0.005 + 1e-9,
        )
        self.assertLessEqual(
            V3_TOP_FIXED_SLOTS["necklace"][1] - V3_TOP_FIXED_SLOTS["glasses"][3],
            0.015 + 1e-9,
        )

    def test_only_necklace_overlaps_top_and_renders_last(self):
        allowed = frozenset(("top", "necklace"))
        for left, right in combinations(V3_TOP_FIXED_SLOTS, 2):
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    overlaps(V3_TOP_FIXED_SLOTS[left], V3_TOP_FIXED_SLOTS[right]),
                    frozenset((left, right)) == allowed,
                )
        self.assertGreater(
            V3_TOP_DRAW_ORDER.index("necklace"), V3_TOP_DRAW_ORDER.index("top")
        )

        top = product_tensor((920, 710), (230, 20, 20, 255))
        necklace = product_tensor((340, 145), (20, 70, 230, 255))
        top_only = self.v3.compose_top(
            1000, 1000, "white", 24, 0, False, top=top
        )[0][0].numpy()
        overlaid = self.v3.compose_top(
            1000, 1000, "white", 24, 0, False, top=top, necklace=necklace
        )[0][0].numpy()
        y, x = 420, 500
        self.assertGreater(top_only[y, x, 0], 0.8)
        self.assertLess(top_only[y, x, 2], 0.2)
        self.assertGreater(overlaid[y, x, 2], 0.8)
        self.assertLess(overlaid[y, x, 0], 0.2)

    def test_pair_earrings_use_separate_symmetric_boxes(self):
        left = V3_TOP_FIXED_SLOTS["earrings_left"]
        right = V3_TOP_FIXED_SLOTS["earrings_right"]
        self.assertFalse(overlaps(left, right))
        self.assertAlmostEqual(left[2] - left[0], right[2] - right[0])
        self.assertAlmostEqual(left[3] - left[1], right[3] - right[1])
        self.assertAlmostEqual((left[0] + left[2] + right[0] + right[2]) / 2, 1.0)

        pair = Image.new("RGBA", (240, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pair)
        draw.ellipse((10, 10, 90, 90), fill=(90, 70, 40, 255))
        draw.ellipse((150, 10, 230, 90), fill=(90, 70, 40, 255))
        placements = self.v3._fixed_earrings(pair, 1536, 2048, 10)
        self.assertEqual([entry[3] for entry in placements], ["earrings L", "earrings R"])
        self.assertEqual(placements[0][2], self.v3._fixed_box("earrings_left", 1536, 2048))
        self.assertEqual(placements[1][2], self.v3._fixed_box("earrings_right", 1536, 2048))

    def test_v3_top_preserves_aspect_and_maximises_each_slot(self):
        samples = {
            "top": ((640, 420), (260, 700), (1000, 100)),
            "hat": ((500, 220),),
            "glasses": ((600, 180),),
            "necklace": ((300, 420),),
        }
        for slot, sizes in samples.items():
            for size in sizes:
                with self.subTest(slot=slot, size=size):
                    cutout = Image.new("RGBA", size, (70, 80, 120, 255))
                    placement, hard = self.v3._fit_fixed_slot(
                        cutout, slot, 1536, 2048, 10
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

    def test_v3_calls_do_not_mutate_v2_layout(self):
        before_boxes = copy.deepcopy(V2_FIXED_SLOTS)
        before_order = tuple(V2_DRAW_ORDER)
        cutout = Image.new("RGBA", (600, 500), (80, 30, 30, 255))
        before_placement = self.v2._fit_fixed_slot(cutout, "top", 1536, 2048, 10)
        self.v3.compose_top(
            768, 1024, "white", 24, 8, True,
            top=product_tensor(), hat=product_tensor((280, 120)),
        )
        self.assertEqual(V2_FIXED_SLOTS, before_boxes)
        self.assertEqual(V2_DRAW_ORDER, before_order)
        self.assertEqual(
            self.v2._fit_fixed_slot(cutout, "top", 1536, 2048, 10),
            before_placement,
        )


if __name__ == "__main__":
    unittest.main()
