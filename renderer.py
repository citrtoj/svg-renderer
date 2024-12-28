import math
import re
import xml.etree.ElementTree
from copy import deepcopy
from typing import Callable

from PIL import Image, ImageDraw

from color import RGBA
from common import Point
from exceptions import InvalidAttributeException
from path import path_data_to_subpaths


def error_on_missing_attribute(tag_name: str):
    """Helper decorator for methods of class Renderer;
    replaces KeyErrors caused by attempts to access missing attributes
    with an `InvalidAttributeException` with a more helpful message.

    Args:
        tag_name (str): name of SVG tag
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except KeyError as e:
                raise InvalidAttributeException(f"Missing <{tag_name}> attribute: {e.args[0]}") from e

        return wrapper

    return decorator


class Renderer:
    """Class used to render a parsed SVG XML to a PIL image object."""

    DEFAULT_ATTRIBUTES = {
        "fill": {"enabled": True, "opacity": 1, "color": (0, 0, 0)},
        "stroke": {"enabled": False, "opacity": 1, "color": (0, 0, 0), "width": 1},
    }

    def reset(self):
        self.image = None
        self.size = (1, 1)  # width, height
        self.viewbox = (0, 0, 1, 1)  # xmin, ymin, width, height

    def __init__(self):
        self.reset()

    def diagonal_length(self) -> float:
        """Return the diagonal length of the image, based on its pixel size.

        Returns:
            float: diagonal length of image
        """
        return math.sqrt(self.size[0] ** 2 + self.size[1] ** 2)

    @staticmethod
    def parse_possible_percentage(value: str, value_on_no_percent: Callable, value_on_percent: Callable):
        """Parse a string which may or may not contain a percentage sign; return transformations on its
        value using given callables.

        Args:
            value (str): string value to parse
            value_on_no_percent (Callable): function whose result will be returned if the string does not
            contain a percentage sign
            value_on_percent (Callable): function whose result will be returned if the string
            contains a percentage sign

        Raises:
            ValueError: if the attribute does not match the pattern for a number followed by a percentage sign

        Returns:
            the result of one of the callables that ends up being called, or None if the value is empty or None
        """
        if not value:
            return None
        match = re.match(r"^([0-9]*.?[0-9])(?P<percent>%)?$", value)
        if not match:
            raise ValueError(f"Invalid attribute value: {value}")
        if match.groupdict()["percent"] is None:
            return value_on_no_percent(value)
        return value_on_percent(value)

    def parse_length(self, value: str):
        """Parse a string which represents a length (for example, for stroke width)."""
        return Renderer.parse_possible_percentage(
            value, lambda x: int(float(x)), lambda x: int(float(x) / 100 * self.diagonal_length())
        )

    @staticmethod
    def parse_opacity(value: str) -> float:
        """Parse a string which represents an opacity (for example, for stroke/fill opacity)."""
        return (
            Renderer.parse_possible_percentage(value, lambda x: float(x), lambda x: float(x) * 100) if value else None
        )

    @staticmethod
    def parse_canvas_size(value: str) -> int:
        """Parse a string which represents a size (for example, for width/height attributes."""
        measurement_unit_values = {"px": 1, "pt": 1.3334, "pc": 16}
        value = value.replace(" ", "").lower()
        match = re.match(r"^-?\d+(\.\d+)?([a-zA-Z0-9]+)?$", value)
        if not match:
            raise ValueError(f"Invalid value for canvas size: {value}")
        measurement_unit = match.group(2)
        if not measurement_unit:
            return int(float(value))
        if measurement_unit in measurement_unit_values:
            return int(float(value[:-2])) * measurement_unit_values[measurement_unit]
        return int(float(value[: -len(measurement_unit)]))

    def update_attributes(self, node, old_attributes: dict) -> dict:
        """Update dictionary of attributes (drawing context) using a node's attributes."""
        attributes = deepcopy(old_attributes)

        def get_opacity(attr_name, fallback_opacity):
            return self.parse_opacity(node.attrib[attr_name]) if attr_name in node.attrib else fallback_opacity

        fill_color = RGBA.from_any(node.attrib.get("fill", ""))
        stroke_color = RGBA.from_any(node.attrib.get("stroke", ""))
        opacity = self.parse_opacity(node.attrib.get("opacity", None)) or 1

        fill_opacity = get_opacity("fill-opacity", fill_color.a if fill_color and fill_color.a else 1)
        stroke_opacity = get_opacity("stroke-opacity", stroke_color.a if stroke_color and stroke_color.a else 1)

        fill_opacity *= opacity
        stroke_opacity *= opacity

        stroke_width = self.parse_length(node.attrib.get("stroke-width", None))

        attributes["fill"].update(
            {
                "enabled": not fill_color.is_none() if fill_color else attributes["fill"]["enabled"],
                "color": fill_color.values[:3]
                if fill_color and not fill_color.is_none()
                else attributes["fill"]["color"],
                "opacity": attributes["fill"]["opacity"] * fill_opacity
                if fill_opacity
                else attributes["fill"]["opacity"],
            }
        )

        attributes["stroke"].update(
            {
                "enabled": not stroke_color.is_none() if stroke_color else attributes["stroke"]["enabled"],
                "color": stroke_color.values[:3]
                if stroke_color and not stroke_color.is_none()
                else attributes["stroke"]["color"],
                "opacity": attributes["stroke"]["opacity"] * stroke_opacity
                if stroke_opacity
                else attributes["stroke"]["opacity"],
                "width": stroke_width if stroke_width is not None else attributes["stroke"]["width"],
            }
        )

        return attributes

    @staticmethod
    def _to_pil_properties(attributes: dict) -> dict:
        """Convert an attributes dictionary into a dictionary suitable for PIL functions
        (such that it can be used as argument with the * syntax).

        Args:
            attributes (dict): dictionary of attributes

        Returns:
            dict: new dictionary with PIL-appropriate values
        """

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
                    *attributes["stroke"]["color"],
                    normalize_opacity(attributes["stroke"]["opacity"]),
                ),
                "width": attributes["stroke"]["width"],
            }
            if attributes["stroke"]["enabled"]
            else {}
        )
        return properties

    def project_point(self, coord: Point) -> Point:
        """Project a point from the SVG viewbox space to the pixel canvas.

        Args:
            coord (Point): point to be projected

        Returns:
            Point: projected point
        """
        return (
            (coord[0] - self.viewbox[0]) / self.viewbox[2] * self.size[0],
            (coord[1] - self.viewbox[1]) / self.viewbox[3] * self.size[1],
        )

    @error_on_missing_attribute("svg")
    def init_svg(self, node) -> None:
        """Initialize image using data from a `<svg>` node in the XML.

        Args:
            node: SVG node
        """
        if self.image is not None:
            raise ValueError("<svg> tag already parsed")

        has_width_height = "width" in node.attrib and "height" in node.attrib
        viewbox_attrs = ["viewbox", "viewBox"]

        if has_width_height:
            width_data = node.attrib["width"]
            height_data = node.attrib["height"]
            w, h = tuple(map(self.parse_canvas_size, [width_data, height_data]))
            if w < 0 or h < 0:
                raise ValueError(f"Invalid <svg> width and height: {w} {h}")
            self.size = (w, h)

        viewbox_data = next((node.attrib[attr] for attr in viewbox_attrs if attr in node.attrib), None)
        if viewbox_data is not None:
            viewbox = tuple(float(x) for x in viewbox_data.split(" ") if x != "")
            if len(viewbox) != 4 or viewbox[2] < 0 or viewbox[3] < 0:
                raise InvalidAttributeException(f"Bad <svg> viewBox: {viewbox_data}")
            self.viewbox = viewbox
            if not has_width_height:
                self.size = (self.viewbox[2], self.viewbox[3])
        else:
            self.viewbox = (0, 0, *self.size)

        if not has_width_height and viewbox_data is None:
            raise InvalidAttributeException("Missing <svg> image dimensions")
        self.image = Image.new("RGBA", self.size)

    @error_on_missing_attribute("ellipse")
    def draw_ellipse(self, node, overlay, attributes: dict) -> Image:
        """Draw an ellipse onto a given overlay with given attributes."""
        cx, cy = (self.viewbox[0] + float(node.attrib["cx"]), self.viewbox[1] + float(node.attrib["cy"]))
        rx, ry = (float(node.attrib["rx"]), float(node.attrib["ry"]))
        colors = self._to_pil_properties(attributes)
        ImageDraw.Draw(overlay).ellipse(
            [self.project_point(x) for x in [(cx - rx, cy - ry), (cx + rx, cy + ry)]],
            **colors,
        )

    @error_on_missing_attribute("rect")
    def draw_rect(self, node, overlay, attributes: dict) -> Image:
        """Draw a rectangle onto a given overlay with given attributes."""
        x, y = (self.viewbox[0] + float(node.attrib["x"]), self.viewbox[1] + float(node.attrib["y"]))
        w, h = (float(node.attrib["width"]), float(node.attrib["height"]))
        rx = ry = 0
        if "rx" in node.attrib:
            rx = ry = float(node.attrib["rx"])
        if "ry" in node.attrib:
            ry = float(node.attrib["ry"])
        colors = self._to_pil_properties(attributes)
        ImageDraw.Draw(overlay).rounded_rectangle(
            list(map(self.project_point, [(x, y), (x + w, y + h)])),
            radius=(self.project_point((rx, 0))[0] + self.project_point((0, ry))[1]) / 2,
            **colors,
        )

    def _draw_polyline(self, overlay, points: list[Point], attributes: dict) -> Image:
        """Helper function to draw a polyline onto a given overlay with given attributes."""
        colors = self._to_pil_properties(attributes)
        if attributes["fill"]["enabled"]:
            ImageDraw.Draw(overlay).polygon(
                points, **{k: v for k, v in colors.items() if k not in {"outline", "width"}}
            )
        if attributes["stroke"]["enabled"]:
            ImageDraw.Draw(overlay).line(points, fill=colors["outline"], width=colors["width"])

    @error_on_missing_attribute("polyline")
    def draw_polyline(self, node, overlay, attributes: dict):
        """Draw a polyline onto a given overlay with given attributes."""
        if not node.attrib.get("points", "").strip():
            raise InvalidAttributeException("Invalid <polyline> tag: missing polyline points")
        points = [tuple(float(y) for y in x.split(",")) for x in node.attrib["points"].strip().split(" ")]
        if any(len(coords) != 2 for coords in points):
            raise InvalidAttributeException(f"Invalid <polyline> tag coordinates: {node.attrib['points']}")
        self._draw_polyline(overlay, list(map(self.project_point, points)), attributes)

    @error_on_missing_attribute("path")
    def draw_path(self, node, overlay, attributes: dict):
        """Draw a path onto a given overlay, using the polyline draw helper, with given attributes."""
        path_data = node.attrib["d"]
        points = path_data_to_subpaths(path_data, path_resolution=int(self.size[0] / 5))
        for subpath in points:
            self._draw_polyline(overlay, list(map(self.project_point, subpath)), attributes)

    def parse_node(self, node, attributes: dict):
        """Parse an SVG XML node, dispatching the drawing operation to the appropriate function
        according to the name of the tag.
        """
        tag = node.tag[len("{http://www.w3.org/2000/svg}") :]
        new_attributes = self.update_attributes(node, attributes)
        overlay = Image.new("RGBA", self.size, (255, 255, 255, 0))
        methods = {
            "ellipse": self.draw_ellipse,
            "rect": self.draw_rect,
            "polyline": self.draw_polyline,
            "path": self.draw_path,
        }

        if tag == "svg":
            overlay = None
            self.init_svg(node)
        elif tag in methods:
            methods[tag](node, overlay, new_attributes)

        if self.image and overlay:
            self.image = Image.alpha_composite(self.image, overlay)

        for child in node:
            self.parse_node(child, new_attributes)

    def render(self, xml: xml.etree.ElementTree):
        """Render a SVG tree onto `self.image`.

        Args:
            xml (xml.etree.ElementTree): XML tree of SVG
        """
        self.reset()
        self.parse_node(xml.getroot(), attributes=self.DEFAULT_ATTRIBUTES)
        return self
