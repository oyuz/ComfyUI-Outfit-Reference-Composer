import unittest

import numpy as np
from PIL import Image, ImageDraw

# Reuse the ComfyUI dependency stubs and node import from the main test module.
import test_nodes as support
from nodes import OutfitReferenceComposer, TEMPLATE


class PantsFootwearSeparationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.node = OutfitReferenceComposer()

    def test_floor_pants_and_footwear_stay_separate_and_complete(self):
        bottom = Image.new("RGBA", (300, 600), (70, 80, 120, 255))
        bottom_layout = {
            "garment_type": "pants",
            "length": "floor",
            "fit": "loose",
        }

        for footwear_type, footwear_size in (
            ("shoes", (300, 240)),
            ("boots", (240, 400)),
        ):
            with self.subTest(footwear_type=footwear_type):
                footwear = Image.new("RGBA", footwear_size, (100, 40, 150, 255))
                shoe_layout = {"footwear_type": footwear_type}
                placements = {
                    "bottom": self.node._bottom_placement(
                        bottom, bottom_layout, 768, 1024
                    ),
                    "shoes": self.node._box_placement(
                        "shoes", footwear, shoe_layout, 768, 1024
                    ),
                }
                packed = self.node._pack_main(
                    placements,
                    {"bottom": bottom_layout, "shoes": shoe_layout},
                    768,
                    1024,
                )
                pant_box = packed["bottom"][0]
                footwear_box = packed["shoes"][0]
                pant_end = pant_box[1] + pant_box[3]

                self.assertGreaterEqual(footwear_box[1] - pant_end, 8)
                self.assertLessEqual(
                    footwear_box[1] + footwear_box[3],
                    round(TEMPLATE["main_bottom_y"] * 1024),
                )
                self.assertLessEqual(
                    support.axis_change(footwear, footwear_box), 1.02
                )

        boot = Image.new("RGBA", (240, 400), (100, 40, 150, 255))
        ImageDraw.Draw(boot).rectangle(
            (0, 300, 239, 399), fill=(190, 35, 35, 255)
        )
        canvas = Image.new("RGBA", (400, 500), "white")
        actual = self.node._composite(canvas, boot, (80, 60, 240, 400))
        self.assertEqual(actual, (80, 60, 320, 460))
        pixels = np.asarray(canvas.convert("RGB"))
        self.assertTrue(np.any(np.all(pixels == (100, 40, 150), axis=-1)))
        self.assertTrue(np.any(np.all(pixels == (190, 35, 35), axis=-1)))


if __name__ == "__main__":
    unittest.main()
