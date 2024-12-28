import re
import json
from typing import Union


def return_none_if_none(f):
    """Decorator used for RGBA properties to make it easier to return None if
    the underlying tuple is None (such as in the case of fill="none")

    Args:
        f (Callable): function that returns a value of the RGBA object
    """

    def wrapper(self, *args, **kwargs):
        if self.is_none():
            return None
        return f(self, *args, **kwargs)

    return wrapper


class RGBA:
    """Class to help use/work with RGBA colors easier."""

    def __init__(self, r: int, g: int, b: int, a: int | None = None, none: bool = False):
        """Initialize a RGBA object from r, g, b, a values, or None if the color is None.

        Args:
            r (int): RGB value for red, from 0 to 255
            g (int): RGB value for red, from 0 to 255
            b (int): RGB value for red, from 0 to 255
            a (float | None, optional): alpha/opacity, from 0 to 255. Defaults to None.
            none (bool, optional): whether the color is None. Defaults to False.

        Raises:
            ValueError: if the values are not in [0, 256)
        """
        if none:
            self.values = None
            return

        if not all(x >= 0 and x <= 256 for x in [r, g, b, a if None else 0]):
            raise ValueError(f"Invalid RGBA color {r} {g} {b} {a}")
        self.values = (r, g, b, a / 256 if a is not None else None)

    @property
    @return_none_if_none
    def r(self):
        """Return the RGB red value for the color.

        Returns:
            int: a number from 0 to 255
        """
        return self.values[0]

    @property
    @return_none_if_none
    def g(self):
        """Return the RGB green value for the color.

        Returns:
            int: a number from 0 to 255
        """
        return self.values[1]

    @property
    @return_none_if_none
    def b(self):
        """Return the RGB blue value for the color.

        Returns:
            int: a number from 0 to 255
        """
        return self.values[2]

    @property
    @return_none_if_none
    def a(self):
        """Return the RGB alpha value for the color.

        Returns:
            int: a number from 0 to 255
        """
        return self.values[3]

    @staticmethod
    def from_hex(hex: str) -> Union["RGBA", None]:
        """Return a RGBA object from a hex string (e.g. "#FFFFFF" or "#FFFFFFFF").

        Returns:
            RGBA | None: a RGBA object if the hex code is valid, or None otherwise
        """
        hex = hex.strip()
        if not hex or hex[0] != "#":
            return None

        if re.match(r"#[0-9a-fA-F]{3}$", hex):
            _, r, g, b = hex[:4]
            hex = f"#{r}{r}{g}{g}{b}{b}"

        if re.match(r"#[0-9a-fA-F]{4}$", hex):
            _, r, g, b, a = hex[:9]
            hex = f"#{r}{r}{g}{g}{b}{b}{a}{a}"

        if re.match(r"#[0-9a-fA-F]{6}$", hex):
            r, g, b = (hex[1:3], hex[3:5], hex[5:7])
            return RGBA(int(r, 16), int(g, 16), int(b, 16))

        if re.match(r"#[0-9a-fA-F]{8}$", hex):
            r, g, b, a = tuple(hex[i : (i + 2)] for i in range(1, 9, 2))
            return RGBA(int(r, 16), int(g, 16), int(b, 16), int(a, 16))

    @staticmethod
    def from_rgba(rgba: str) -> Union["RGBA", None]:
        """Return a RGBA object from a rgba string (e.g. "rgb(0,255,0)" or "rgba(0,255,0,0.25)").

        Returns:
            RGBA | None: a RGBA object if the string is valid, or None otherwise
        """
        rgba = rgba.strip().lower().replace(" ", "")
        match = re.match(r"^rgb(?P<is_a>a)?\((?P<r>\d{1,3}),(?P<g>\d{1,3}),(?P<b>\d{1,3})(,(?P<a>\d{1,3}))?\)$", rgba)
        if match:
            match_groups = match.groupdict()
            if (match_groups["is_a"] is not None) != (match_groups["a"] is not None):
                raise ValueError(f"Invalid rgba value: {rgba}")

            return RGBA(
                int(match_groups["r"]),
                int(match_groups["g"]),
                int(match_groups["b"]),
                float(match_groups["a"]) * 256 if match_groups.get("a", None) else None,
            )

    @staticmethod
    def from_name(value: str) -> Union["RGBA", None]:
        """Return a RGBA object from a color name (e.g. "blue" or "darkslategrey").

        Returns:
            RGBA | None: a RGBA object if the name is valid, or None otherwise
        """
        value = value.strip().lower()
        with open("assets/css_color_names.json", "r") as f:
            color_names = json.load(f)
            if value not in color_names:
                return None
            return RGBA.from_hex(color_names[value])

    @staticmethod
    def from_none(value: str) -> Union["RGBA", None]:
        """Return a RGBA object from the "none" string.

        Returns:
            RGBA | None: a RGBA object if the string is valid, or None otherwise
        """
        if value.strip().lower() == "none":
            return RGBA(0, 0, 0, 0, none=True)

    @staticmethod
    def from_any(value: str | None) -> Union["RGBA", None]:
        """Return a RGBA object from any of the aforementioned methods,
        or None if no methods matched.

        Returns:
            RGBA | None: a RGBA object if the string is valid, or None otherwise
        """
        if not value:
            return None
        color = RGBA.from_hex(value)
        color = RGBA.from_rgba(value) if color is None else color
        color = RGBA.from_name(value) if color is None else color
        color = RGBA.from_none(value) if color is None else color
        return color

    def is_none(self):
        """Return whether the color represents a "none" attribute (e.g. fill="none").

        Returns:
            bool: whether the color is "none"
        """
        return self.values is None

    def __str__(self) -> str:
        return str(self.values)
