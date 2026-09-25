"""Turn the supplied poster's headline letterforms into small text fonts.

Usage: uv run --with fonttools --with opencv-python-headless --with pillow \
    python scripts/build-poster-lettering.py /path/to/the/supplied/poster.jpg

Only individual headline glyph outlines are retained, never raster images or
whole-line picture glyphs. These are limited lettering subsets for this poster,
not the original designer's complete typeface. Body text uses the licensed Anton
font. Keep the source poster outside the public site.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


def region(image, box, dark=False):
    scale = image.shape[1] / 1367
    x1, y1, x2, y2 = [round(value * scale) for value in box]
    crop = image[y1:y2, x1:x2]
    mask = ((crop[:, :, 1] < 75) if dark else (crop[:, :, 1] > 75)).astype("uint8")
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    parts = sorted(
        [(index, stat) for index, stat in enumerate(stats[1:], 1) if stat[4] > 50],
        key=lambda item: item[1][0],
    )
    return labels, parts


def glyph_name(char):
    return "space" if char == " " else "uni%04X" % ord(char)


def make_font(image, family, filename, specifications, small_i=False):
    outlines = {}
    metrics = {}
    cmap = {}
    pair_values = defaultdict(list)
    # Spaces remain real selectable characters; their visible spacing is fitted
    # by the adjacent kerning pair, just like spacing inside the poster's words.
    space_advance = 80
    for specification in specifications:
        box, text, cap_height, baseline, dark = specification
        labels, parts = region(image, box, dark)
        nonspaces = text.replace(" ", "")
        assert len(parts) == len(nonspaces), (family, text, len(parts), len(nonspaces))
        scale = 1000 / cap_height
        letters = []
        previous = None
        for char, (component, (x, y, width, height, area)) in zip(nonspaces, parts):
            name = "I.small" if small_i and char == "I" and previous == "D" else glyph_name(char)
            if name not in outlines:
                isolated = (labels[y:y + height, x:x + width] == component).astype("uint8")
                contours, _ = cv2.findContours(isolated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                pen = TTGlyphPen(None)
                for contour in contours:
                    if abs(cv2.contourArea(contour)) < 1:
                        continue
                    points = cv2.approxPolyDP(contour, 0.25, True).reshape(-1, 2)
                    coordinates = [(float(px) * scale, (baseline - y - float(py)) * scale) for px, py in points]
                    pen.moveTo(coordinates[0])
                    for point in coordinates[1:]:
                        pen.lineTo(point)
                    pen.closePath()
                outlines[name] = pen.glyph()
                metrics[name] = (round(width * scale), 0)
                if name != "I.small":
                    cmap[ord(char)] = name
            letters.append((name, x * scale))
            previous = char

        pointer = 0
        for index, char in enumerate(text[:-1]):
            if char == " ":
                continue
            current_name, current_x = letters[pointer]
            pointer += 1
            next_name, next_x = letters[pointer]
            if text[index + 1] == " ":
                pair = (current_name, "space")
                adjustment = next_x - current_x - metrics[current_name][0] - space_advance
            else:
                pair = (current_name, next_name)
                adjustment = next_x - current_x - metrics[current_name][0]
            pair_values[pair].append(round(adjustment))

    empty = TTGlyphPen(None).glyph()
    outlines[".notdef"] = empty
    outlines["space"] = empty
    metrics[".notdef"] = (500, 0)
    metrics["space"] = (space_advance, 0)
    cmap[32] = "space"
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder([".notdef", *[name for name in outlines if name != ".notdef"]])
    builder.setupCharacterMap(cmap)
    builder.setupGlyf(outlines)
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=1100, descent=-100)
    builder.setupNameTable({
        "familyName": family,
        "styleName": "Regular",
        "uniqueFontIdentifier": family + " 1.0",
        "fullName": family,
        "psName": family.replace(" ", ""),
        "version": "Version 1.0",
        "description": "Limited headline lettering reconstructed from the supplied Sagra poster.",
    })
    builder.setupOS2(sTypoAscender=1100, sTypoDescender=-100, usWinAscent=1100, usWinDescent=100, sCapHeight=1000)
    builder.setupPost()
    features = "languagesystem DFLT dflt;\n"
    if small_i:
        features += "feature calt { sub uni0044 uni0049' by I.small; } calt;\n"
    features += "feature kern {\n"
    for (left, right), values in pair_values.items():
        features += "  pos %s %s %d;\n" % (left, right, round(median(values)))
    features += "} kern;\n"
    addOpenTypeFeaturesFromString(builder.font, features)
    destination = Path(__file__).resolve().parents[1] / "assets/fonts" / filename
    builder.save(destination)
    print(f"Built {destination.name}: {len(cmap)} supported characters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poster", type=Path)
    arguments = parser.parse_args()
    image = np.asarray(Image.open(arguments.poster).convert("RGB"))
    assert image.shape[:2] == (2732, 2048), "Use the original 2048 by 2732 poster."
    make_font(image, "Sagra Poster Wordmark", "sagra-poster-wordmark.ttf", [
        ((120, 110, 1245, 244), "VIa SAGRA DI TORONTO", 174, 191, False),
    ], small_i=True)
    make_font(image, "Sagra Poster Display", "sagra-poster-display.ttf", [
        ((180, 370, 1185, 468), "SUNDAY OCTOBER 25", 120, 135, True),
        ((180, 473, 1185, 555), "THE COLUMBUS CENTRE", 104, 113, True),
    ])


if __name__ == "__main__":
    main()
