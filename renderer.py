import re
import math
import xml.etree.ElementTree
from color import RGBA
from PIL import Image, ImageDraw
from copy import deepcopy


class Renderer:
    DEFAULT_ATTRIBUTES = {
        "fill": {"opacity": 1, "color": (255, 255, 255)},
        "stroke": {"opacity": 1, "color": (0, 0, 0), "width": 5},
    }

    def reset(self):
        self.image = None
        self.size = tuple()
        self.point = [0.0, 0.0]

    def __init__(self):
        self.reset()

    def _diagonal_length(self):
        if not self.image:
            raise ValueError("No image found!")
        return math.sqrt(self.size[0] ** 2 + self.size[1] ** 2)

    @staticmethod
    def _parse_percent(value_str: str, value_on_no_percent, value_on_percent):
        match = re.match(r"^([0-9]*.?[0-9])(?P<percent>%)?$", value_str)
        if match:
            if match.groupdict()["percent"] is None:
                return value_on_no_percent(value_str)
            return value_on_percent(value_str)

    def _parse_length(self, value: str):
        return Renderer._parse_percent(
            value, lambda x: int(float(x)), lambda x: int(float(x) / 100 * self._diagonal_length())
        )

    @staticmethod
    def _parse_opacity(value: str) -> int:
        return Renderer._parse_percent(value, lambda x: float(x), lambda x: float(x) * 100)

    def _update_attributes(self, node, old_attributes=None):
        if not old_attributes:
            old_attributes = self.DEFAULT_ATTRIBUTES
        attributes = deepcopy(old_attributes)

        fill = RGBA.from_any(node.attrib["fill"] if "fill" in node.attrib else "")
        stroke = RGBA.from_any(node.attrib["stroke"] if "stroke" in node.attrib else "")

        opacity = self._parse_opacity(node.attrib["opacity"]) if "opacity" in node.attrib else None
        fill_opacity = (
            self._parse_opacity(node.attrib["fill-opacity"])
            if "fill-opacity" in node.attrib
            else (fill.a if fill and fill.a else None)
        )
        stroke_opacity = (
            self._parse_opacity(node.attrib["stroke-opacity"])
            if "stroke-opacity" in node.attrib
            else (stroke.a if stroke and stroke.a else None)
        )
        if opacity:
            fill_opacity = (fill_opacity if fill_opacity else attributes["fill"]["opacity"]) * opacity
            stroke_opacity = (stroke_opacity if stroke_opacity else attributes["stroke"]["opacity"]) * opacity
        stroke_width = self._parse_length(node.attrib["stroke-width"]) if "stroke-opacity" in node.attrib else None

        if fill:
            attributes["fill"]["color"] = fill.values[:3]
        if stroke:
            attributes["stroke"]["color"] = stroke.values[:3]
        if fill_opacity:
            attributes["fill"]["opacity"] *= fill_opacity
        if stroke_opacity:
            attributes["stroke"]["opacity"] *= stroke_opacity
        if stroke_width:
            attributes["stroke"]["width"] = stroke_width

        return attributes

    @staticmethod
    def _to_pil_colors(attributes):
        def normalize_opacity(opacity):
            return min(255, int(opacity * 256))

        return {
            "fill": (*attributes["fill"]["color"], normalize_opacity(attributes["fill"]["opacity"])),
            "stroke": (*attributes["stroke"]["color"], normalize_opacity(attributes["stroke"]["opacity"])),
        }

    def _draw_svg(self, node, attributes) -> None:
        if self.image is not None:
            raise ValueError("<svg> tag already parsed")
        if "width" in node.attrib and "height" in node.attrib:
            w = int(node.attrib["width"])
            h = int(node.attrib["height"])
            size = (w, h)
        else:
            raise ValueError("Could not find dimensions of image. Aborting")
        self.image = Image.new("RGBA", size)

    def _draw_ellipse(self, node, attributes) -> Image:
        # TODO: check existence of cx, cy, rx and ry
        cx = self.point[0] + float(node.attrib["cx"])
        cy = self.point[1] + float(node.attrib["cy"])
        rx = float(node.attrib["rx"])
        ry = float(node.attrib["ry"])
        overlay = Image.new("RGBA", self.image.size, (255, 255, 255, 0))
        colors = self._to_pil_colors(attributes)
        ImageDraw.Draw(overlay).ellipse(
            [(cx - rx, cy - ry), (cx + rx, cy + ry)],
            fill=colors["fill"],
            outline=colors["stroke"],
            width=attributes["stroke"]["width"],
        )
        return overlay

    def _draw_rect(self, node, attributes) -> Image:
        x = self.point[0] + float(node.attrib["x"])
        y = self.point[1] + float(node.attrib["y"])
        w = float(node.attrib["width"])
        h = float(node.attrib["height"])
        rx = 0
        ry = 0
        if "rx" in node.attrib:
            rx = float(node.attrib["rx"])
            ry = float(node.attrib["rx"])
        if "ry" in node.attrib:
            ry = float(node.attrib["ry"])
        overlay = Image.new("RGBA", self.image.size, (255, 255, 255, 0))
        colors = self._to_pil_colors(attributes)
        ImageDraw.Draw(overlay).rounded_rectangle(
            [(x, y), (x + w, y + h)],
            radius=(rx + ry) / 2,
            fill=colors["fill"],
            outline=colors["stroke"],
            width=attributes["stroke"]["width"],
        )
        return overlay

    def parse_node(self, node, attributes):
        tag = node.tag[len("{http://www.w3.org/2000/svg}") :]
        print(f"Processing tag: {tag}")
        overlay = None

        new_attributes = self._update_attributes(node, attributes)

        if tag == "svg":
            self._draw_svg(node, attributes=new_attributes)
        elif tag == "ellipse":
            overlay = self._draw_ellipse(node, attributes=new_attributes)
        elif tag == "rect":
            overlay = self._draw_rect(node, attributes=new_attributes)

        if self.image and overlay:
            self.image = Image.alpha_composite(self.image, overlay)

        for child in node:
            self.parse_node(child, new_attributes)

    def render(self, xml: xml.etree.ElementTree):
        self.reset()
        self.parse_node(xml.getroot(), attributes=self.DEFAULT_ATTRIBUTES)
        return self
