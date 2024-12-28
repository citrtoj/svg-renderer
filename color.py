import re
from typing import Union


def return_none_if_empty(f):
    def wrapper(self, *args, **kwargs):
        if self.is_none():
            return None
        return f(self, *args, **kwargs)

    return wrapper


class RGBA:
    def __init__(self, r: int, g: int, b: int, a: float | None = None, none: bool = False):
        if none:
            self.values = None
        else:
            self.values = (r, g, b, a / 256 if a is not None else None)

    @property
    @return_none_if_empty
    def r(self):
        return self.values[0]

    @property
    @return_none_if_empty
    def g(self):
        return self.values[1]

    @property
    @return_none_if_empty
    def b(self):
        return self.values[2]

    @property
    @return_none_if_empty
    def a(self):
        return self.values[3]

    @staticmethod
    def from_hex(hex: str) -> Union["RGBA", None]:
        hex = hex.strip()
        if not hex or hex[0] != "#":
            return
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
    def from_none(value: str) -> Union["RGBA", None]:
        if value.strip().lower() == "none":
            return RGBA(0, 0, 0, 0, none=True)

    @staticmethod
    def from_any(value: str | None) -> Union["RGBA", None]:
        if not value:
            return None
        color = RGBA.from_hex(value)
        color = RGBA.from_rgba(value) if color is None else color
        color = RGBA.from_none(value) if color is None else color
        return color

    def is_none(self):
        return self.values is None

    def __str__(self) -> str:
        return str(self.values)


if __name__ == "__main__":
    print(RGBA.from_rgba("rgba(255, 255, 255)"))
