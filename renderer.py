import re
import math
import xml.etree.ElementTree
from color import RGBA
from PIL import Image, ImageDraw
from copy import deepcopy
from exceptions import InvalidAttributeException
from path import path_data_to_points


def error_on_missing_attribute(tag_name):
    def decorator(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except KeyError as e:
                raise InvalidAttributeException(f"Missing <{tag_name}> attribute: {e.args[0]}")

        return wrapper

    return decorator


class Renderer:
    DEFAULT_ATTRIBUTES = {
        "fill": {"enabled": True, "opacity": 1, "color": (0, 0, 0)},
        "stroke": {"enabled": False, "opacity": 1, "color": (0, 0, 0), "width": 1, "linecap": "butt"},
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

        fill_color = RGBA.from_any(node.attrib["fill"] if "fill" in node.attrib else "")
        stroke_color = RGBA.from_any(node.attrib["stroke"] if "stroke" in node.attrib else "")

        opacity = self._parse_opacity(node.attrib["opacity"]) if "opacity" in node.attrib else None
        fill_opacity = (
            self._parse_opacity(node.attrib["fill-opacity"])
            if "fill-opacity" in node.attrib
            else (fill_color.a if fill_color and fill_color.a else None)
        )
        stroke_opacity = (
            self._parse_opacity(node.attrib["stroke-opacity"])
            if "stroke-opacity" in node.attrib
            else (stroke_color.a if stroke_color and stroke_color.a else None)
        )
        if opacity:
            fill_opacity = (fill_opacity if fill_opacity else attributes["fill"]["opacity"]) * opacity
            stroke_opacity = (stroke_opacity if stroke_opacity else attributes["stroke"]["opacity"]) * opacity
        stroke_width = self._parse_length(node.attrib["stroke-width"]) if "stroke-width" in node.attrib else None

        if fill_color:
            if fill_color.is_empty():
                attributes["fill"]["enabled"] = False
            else:
                attributes["fill"]["color"] = fill_color.values[:3]
                attributes["fill"]["enabled"] = True
        if stroke_color:
            if stroke_color.is_empty():
                attributes["stroke"]["enabled"] = False
            else:
                attributes["stroke"]["color"] = stroke_color.values[:3]
                attributes["stroke"]["enabled"] = True
        if fill_opacity:
            attributes["fill"]["opacity"] *= fill_opacity
        if stroke_opacity:
            attributes["stroke"]["opacity"] *= stroke_opacity
        if stroke_width:
            attributes["stroke"]["width"] = stroke_width

        return attributes

    @staticmethod
    def _to_pil_properties(attributes):
        def normalize_opacity(opacity):
            return min(255, int(opacity * 256))

        properties = (
            {
                "fill": (*attributes["fill"]["color"], normalize_opacity(attributes["fill"]["opacity"])),
            }
            if attributes["fill"]["enabled"]
            else {}
        )
        properties.update(
            {
                "outline": (
                    *attributes["stroke"]["color"],  # RGB values
                    normalize_opacity(attributes["stroke"]["opacity"]),
                ),
                "width": attributes["stroke"]["width"],
            }
            if attributes["stroke"]["enabled"]
            else {}
        )
        print(properties)
        return properties

    @error_on_missing_attribute("svg")
    def _draw_svg(self, node) -> None:
        if self.image is not None:
            raise ValueError("<svg> tag already parsed")
        if "width" in node.attrib and "height" in node.attrib:
            w = int(node.attrib["width"])
            h = int(node.attrib["height"])
            size = (w, h)
        else:
            raise InvalidAttributeException("Missing <svg> image dimensions")
        self.image = Image.new("RGBA", size)

    @error_on_missing_attribute("ellipse")
    def _draw_ellipse(self, node, attributes) -> Image:
        # TODO: check existence of cx, cy, rx and ry
        cx = self.point[0] + float(node.attrib["cx"])
        cy = self.point[1] + float(node.attrib["cy"])
        rx = float(node.attrib["rx"])
        ry = float(node.attrib["ry"])
        overlay = Image.new("RGBA", self.image.size, (255, 255, 255, 0))
        colors = self._to_pil_properties(attributes)
        ImageDraw.Draw(overlay).ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], **colors)
        return overlay

    @error_on_missing_attribute("rect")
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
        colors = self._to_pil_properties(attributes)
        ImageDraw.Draw(overlay).rounded_rectangle([(x, y), (x + w, y + h)], radius=(rx + ry) / 2, **colors)

        return overlay

    def _draw_polyline_points(self, overlay, points, attributes):
        colors = self._to_pil_properties(attributes)

        if attributes["fill"]["enabled"]:
            ImageDraw.Draw(overlay).polygon(
                points, **{k: v for k, v in colors.items() if k not in {"outline", "width"}}
            )
        if attributes["stroke"]["enabled"]:
            ImageDraw.Draw(overlay).line(points, fill=colors["outline"], width=colors["width"])

        return overlay

    @error_on_missing_attribute("polyline")
    def _draw_polyline(self, node, attributes):
        if "points" not in node.attrib or not node.attrib["points"].strip():
            raise InvalidAttributeException("Invalid <polyline> tag: missing polyline points")

        points = [tuple(int(y) for y in x.split(",")) for x in node.attrib["points"].strip().split(" ")]
        print(points)
        if any(len(coords) != 2 for coords in points):
            raise InvalidAttributeException(f"Invalid <polyline> tag coordinates: {node.attrib['points']}")

        overlay = Image.new("RGBA", self.image.size, (255, 255, 255, 0))
        return self._draw_polyline_points(overlay, points, attributes)

    @error_on_missing_attribute("path")
    def _draw_path(self, node, attributes):
        path_data = node.attrib["d"]

        points = path_data_to_points(path_data)
        overlay = Image.new("RGBA", self.image.size, (255, 255, 255, 0))
        for subpath in points:
            overlay = self._draw_polyline_points(overlay, subpath, attributes)

        return overlay

    def parse_node(self, node, attributes):
        tag = node.tag[len("{http://www.w3.org/2000/svg}") :]
        print(f"Processing tag: {tag}")
        overlay = None

        new_attributes = self._update_attributes(node, attributes)

        match tag:
            case "svg":
                self._draw_svg(node)
            case "ellipse":
                overlay = self._draw_ellipse(node, attributes=new_attributes)
            case "rect":
                overlay = self._draw_rect(node, attributes=new_attributes)
            case "polyline":
                overlay = self._draw_polyline(node, attributes=new_attributes)
            case "path":
                overlay = self._draw_path(node, attributes=new_attributes)

        if self.image and overlay:
            self.image = Image.alpha_composite(self.image, overlay)

        for child in node:
            self.parse_node(child, new_attributes)

    def render(self, xml: xml.etree.ElementTree):
        self.reset()
        self.parse_node(xml.getroot(), attributes=self.DEFAULT_ATTRIBUTES)
        return self
