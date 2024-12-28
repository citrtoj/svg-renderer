import math
import re
from copy import deepcopy
from enum import Enum
from functools import partial

from common import Point

type PathCommand = tuple[str, tuple[float]]


def reflection(point: Point, current_point: Point) -> Point:
    """Return the reflection of a point in relation to a current point."""
    return (2 * current_point[0] - point[0], 2 * current_point[1] - point[1])


def interpolate(interpolation_callable, resolution: int):
    """Generate n points between 0 and 1 and map them to a callable interpolation function;
    return the resulted array.

    Args:
        interpolation_callable (Callable): function that should be called for each value
        resolution (int): number of values

    Returns:
        list: list of mapped values
    """
    return list(map(interpolation_callable, (i / (resolution - 1) for i in range(resolution))))


def cubic_bezier_interpolation(
    point_0: Point, control_point_0: Point, control_point_1: Point, point_1: Point, t: float
) -> Point:
    """Calculate a point on a cubic Bezier curve at parameter t."""
    x = (
        (1 - t) ** 3 * point_0[0]
        + 3 * (1 - t) ** 2 * t * control_point_0[0]
        + 3 * (1 - t) * t**2 * control_point_1[0]
        + t**3 * point_1[0]
    )
    y = (
        (1 - t) ** 3 * point_0[1]
        + 3 * (1 - t) ** 2 * t * control_point_0[1]
        + 3 * (1 - t) * t**2 * control_point_1[1]
        + t**3 * point_1[1]
    )
    return (x, y)


def quadratic_bezier_interpolation(point_0: tuple, control_point: tuple, point_1: tuple, t: float) -> Point:
    """Calculate a point on a quadratic Bezier curve at parameter t."""
    x = (1 - t) ** 2 * point_0[0] + 2 * (1 - t) * t * control_point[0] + t**2 * point_1[0]
    y = (1 - t) ** 2 * point_0[1] + 2 * (1 - t) * t * control_point[1] + t**2 * point_1[1]
    return (x, y)


def cubic_bezier(point_0: Point, control_point_0: Point, control_point_1: Point, point_1: Point, n: int):
    """
    Calculate n points along a cubic Bezier curve.

    Args:
        point_0 (Point): Start point (x0, y0).
        point_1 (Point): End point (x1, y1).
        control_point_0 (Point): First control point (cx0, cy0).
        control_point_1 (Point): Second control point (cx1, cy1).
        n (int): Number of intermediary points to calculate.

    Returns:
        list[Point]: List of points along the curve.
    """

    return interpolate(partial(cubic_bezier_interpolation, point_0, control_point_0, control_point_1, point_1), n)


def quadratic_bezier(point_0: Point, control_point: Point, point_1: Point, n: int):
    """
    Calculate n points along a quadratic Bézier curve.

    Args:
        point_0 (Point): Start point (x0, y0).
        control_point (Point): Control point (cx, cy).
        point_1 (Point): End point (x1, y1).
        n (int): Number of intermediary points to calculate.

    Returns:
        list[Point]: List of points along the curve.
    """
    return interpolate(partial(quadratic_bezier_interpolation, point_0, control_point, point_1), n)


def dot_product(u: Point, v: Point) -> float:
    """Calculate dot product between 2 2D vectors."""
    return u[0] * v[0] + u[1] * v[1]


def norm(v: Point) -> float:
    """Calculate Euclidean norm of a 2D vector."""
    return math.sqrt(dot_product(v, v))


def angle(u: Point, v: Point) -> float:
    """Calculate angle between 2 2D vectors."""
    value = dot_product(u, v) / (norm(u) * norm(v))
    # fix for precision
    if value < -1.0:
        value = -1.0
    elif value > 1.0:
        value = 1.0
    cosine = math.acos(value)
    sign = 1 if u[0] * v[1] - u[1] * v[0] >= 0 else -1
    return sign * cosine


def arc_interpolation(center: Point, rx: float, ry: float, phi: float, theta: float) -> Point:
    """Return point on arc at angle theta along the arc, using center parametrization.

    Args:
        center (Point): center point of ellipse
        rx (float): x radius of ellipse
        ry (float): y radius of ellipse
        phi (float): in radians, x-axis-rotation of ellipse
        theta (float): in radians, angle along the arc

    Returns:
        Point: point along the arc
    """
    return (
        math.cos(phi) * rx * math.cos(theta) - math.sin(phi) * ry * math.sin(theta) + center[0],
        math.sin(phi) * rx * math.cos(theta) + math.cos(phi) * ry * math.sin(theta) + center[1],
    )


def arc(
    rx: float,
    ry: float,
    x_axis_rotation: float,
    large_arc_flag: float,
    sweep_flag: float,
    start_point: Point,
    end_point: Point,
    resolution: int,
) -> list[Point]:
    """Generate n points along an arc specified through endpoint parametrization, by converting it
    to center parametrization then using interpolation with the new parameters.

    Args:
        rx (float): x radius of underlying ellipse
        ry (float): y radius of underlying ellipse
        x_axis_rotation (float): in radians, how much is ellipse rotated on the x axis
        large_arc_flag (int): either 0.0 or 1.0
        sweep_flag (int): either 0.0 or 1.0
        start_point (Point): beginning of arc
        end_point (Point): end of arc
        resolution (int): number of segments that the arc is made of

    Returns:
        list[Point]: list of points describing segments making up the arc
    """
    if rx == 0 or ry == 0 or start_point == end_point:
        return [start_point, end_point]

    # start conversion to center parameterization

    x_axis_rotation = math.fmod(x_axis_rotation, 360.0)
    x_axis_rotation_rad = math.radians(x_axis_rotation)
    cos_rotation = math.cos(x_axis_rotation_rad)
    sin_rotation = math.sin(x_axis_rotation_rad)

    dx = (start_point[0] - end_point[0]) / 2
    dy = (start_point[1] - end_point[1]) / 2

    start_point_p: Point = (
        cos_rotation * dx + sin_rotation * dy,
        -sin_rotation * dx + cos_rotation * dy,
    )

    big_alpha = start_point_p[0] ** 2 / rx**2 + start_point_p[1] ** 2 / ry**2
    if big_alpha > 1:
        rx = math.sqrt(big_alpha) * rx
        ry = math.sqrt(big_alpha) * ry

    c_prime_coef_radicand = (rx**2 * ry**2 - rx**2 * start_point_p[1] ** 2 - ry**2 * start_point_p[0] ** 2) / (
        rx**2 * start_point_p[1] ** 2 + ry**2 * start_point_p[0] ** 2
    )
    # fix for precision
    if c_prime_coef_radicand < 0:
        c_prime_coef_radicand = 0

    c_prime_coef = math.sqrt(c_prime_coef_radicand)

    if large_arc_flag == sweep_flag:
        c_prime_coef = -c_prime_coef

    c_prime: Point = (c_prime_coef * rx * start_point_p[1] / ry, -c_prime_coef * ry * start_point_p[0] / rx)

    center: Point = (
        cos_rotation * c_prime[0] - sin_rotation * c_prime[1] + ((start_point[0] + end_point[0]) / 2),
        sin_rotation * c_prime[0] + cos_rotation * c_prime[1] + ((start_point[1] + end_point[1]) / 2),
    )

    intermediary_point: Point = ((start_point_p[0] - c_prime[0]) / rx, (start_point_p[1] - c_prime[1]) / ry)

    theta_1 = angle((1.0, 0.0), intermediary_point)

    delta_theta_angle = angle(
        intermediary_point, ((-start_point_p[0] - c_prime[0]) / rx, (-start_point_p[1] - c_prime[1]) / ry)
    )
    delta_theta_angle_degrees = math.degrees(delta_theta_angle)
    delta_theta = delta_theta_angle_degrees % 360.0
    if sweep_flag == 0 and delta_theta > 0:
        delta_theta = delta_theta - 360
    elif sweep_flag == 1 and delta_theta < 0:
        delta_theta = delta_theta + 360
    delta_theta = math.radians(delta_theta)

    return [
        arc_interpolation(center, rx, ry, x_axis_rotation_rad, t)
        for t in (theta_1 + i * delta_theta / (resolution - 1) for i in range(resolution))
    ]


class PathCommandTypes(Enum):
    C = 6
    S = 4
    L = 2
    H = 1
    V = 1
    Z = 0
    M = 2
    Q = 4
    T = 2
    A = 7

    def check_if_valid(command: str) -> bool:
        if command.upper() not in PathCommandTypes.__members__:
            raise ValueError(f"Invalid command {command}; not in list {PathCommandTypes.__members__.keys()}")

    def get_length(command: str) -> int:
        """Return the length of a path command type, if valid.

        Args:
            command (str): command type

        Raises:
            ValueError: if the command is not one single letter
            ValueError: if the command is not in the list

        Returns:
            int: length of arguments of said command type
        """
        return PathCommandTypes.__members__[command.upper()].value


def parse_path_commands(path_data: str) -> list[PathCommand]:
    """Parse a `d` string from a `<path>` SVG tag into a list of relative and absolute
    path commands.

    Args:
        path_data (str): string with path data

    Raises:
        ValueError: if a command has the wrong number of arguments, or is not otherwise a valid command

    Returns:
        list[PathCommand]: list of parsed path commands with their respective values
    """
    commands = []
    pattern = re.compile(r"([a-zA-Z])|([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    matches = pattern.findall(path_data)
    current_command_type = None
    current_values = []

    def add_command(command_type, values):
        PathCommandTypes.check_if_valid(command_type)
        command_args_count = PathCommandTypes.get_length(command_type)
        if command_args_count == 0 and len(values) == 0:
            commands.append((command_type, tuple()))
        elif command_args_count > 0 and len(values) % command_args_count == 0:
            for i in range(0, len(values), command_args_count):
                commands.append(
                    (
                        command_type,
                        tuple(values[i : i + command_args_count]),
                    )
                )
        else:
            raise ValueError(f'Wrong length of arguments ({len(values)}) for command "{command_type} {values}"')

    for match in matches:
        if match[0]:
            if current_command_type is not None:
                add_command(current_command_type, current_values)
            current_command_type = match[0]
            current_values = []
        elif match[1]:
            current_values.append(float(match[1]))

    if current_command_type is not None:
        add_command(current_command_type, current_values)

    return commands


def relative_to_absolute(command: PathCommand, current_point: Point) -> PathCommand:
    """Convert a relative path command into an absolute one using a given current point.

    Args:
        command (PathCommand): command to be converted
        current_point (Point): current point

    Returns:
        PathCommand: absolute path command
    """
    if command[0].upper() == command[0]:
        # already absolute
        return command

    values = command[1]
    new_command_type = command[0].upper()
    new_values = []

    match new_command_type:
        case "H":
            new_values.append(current_point[0] + values[0])
        case "V":
            new_values.append(current_point[1] + values[0])
        case "A":
            new_values = [*values[:5], current_point[0] + values[5], current_point[1] + values[6]]
        case _:
            for i, value in enumerate(values):
                new_values.append(current_point[i % 2] + value)

    return (new_command_type, tuple(new_values))


def path_commands_to_points(commands: list[PathCommand], resolution: int) -> list[list[Point]]:
    """Go through a list of parsed path commands and turn them into lists of subpaths
    (list of lists of points).

    Args:
        commands (list[PathCommand]): path commands, relative or absolute
        resolution (int): number of segments a curved line, such as a Bezier curve or an arc,
        should be made up of

    Returns:
        list[list[Point]]: list of subpaths made up of simple line segments
    """
    current_point: Point = (0, 0)
    paths: list[list[Point]] = []
    subpath_points: list[Point] = []

    last_control_point = None  # for S/T commands

    def end_subpath():
        nonlocal subpath_points, current_point, paths
        if len(subpath_points) >= 2:
            paths.append(deepcopy(subpath_points))
        subpath_points.clear()
        subpath_points.append(current_point)

    def values_to_points(values: tuple[float]):
        return list(zip(values[0::2], values[1::2]))

    for i, command in enumerate(commands):
        absolute_command = relative_to_absolute(command, current_point)
        command_type, values = absolute_command

        match command_type:
            case "M":
                point = (values[0], values[1])
                current_point = point
                if i == 0 or commands[i - 1][0].upper() != "M":
                    # first moveto in the sequence
                    end_subpath()
                subpath_points.append(point)
            case "Z":
                # add closing line manually
                subpath_points.append(subpath_points[0])
                current_point = subpath_points[0]
                # close the subpath
                end_subpath()
            case "C" | "Q":
                command_points = values_to_points(values)
                subpath_points.extend(
                    cubic_bezier(current_point, *command_points, resolution)
                    if command_type == "C"
                    else quadratic_bezier(current_point, *command_points, resolution)
                )
                last_control_point = command_points[-2]
            case "S" | "T":
                first_control_point = current_point
                prev_command_allowed_types = {"S", "C"} if command_type == "S" else {"Q", "T"}
                if (
                    i > 0
                    and commands[i - 1][0].upper() in prev_command_allowed_types
                    and last_control_point is not None
                ):
                    first_control_point = reflection(last_control_point, current_point)
                command_points = values_to_points(values)

                subpath_points.extend(
                    cubic_bezier(current_point, first_control_point, *command_points, resolution)
                    if absolute_command[0] == "S"
                    else quadratic_bezier(current_point, first_control_point, *command_points, resolution)
                )
                last_control_point = command_points[0]

            case "V":
                subpath_points.append((current_point[0], values[0]))
            case "H":
                subpath_points.append((values[0], current_point[1]))
            case "L":
                subpath_points.append((values[0], values[1]))
            case "A":
                arc_points = arc(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    current_point,
                    (values[5], values[6]),
                    resolution,
                )
                subpath_points.extend(arc_points)
                subpath_points.append((values[5], values[6]))
            case _:
                pass

        if len(subpath_points) > 0:
            current_point = subpath_points[-1]

    if len(subpath_points) > 0:
        end_subpath()

    return paths


def path_data_to_subpaths(data: str, path_resolution=1000) -> list[list[Point]]:
    """Return list of subpaths (lists of points) from a path data string resulted from the
    `d` attribute of `<path>` tags in SVGs.

    Args:
        data (str): path data string
        path_resolution (int, optional): amount of points to use when drawing curves. Defaults to 1000.

    Returns:
        list[Point]: list of absolute points to draw on the image, in given order
    """
    commands = parse_path_commands(data)
    return path_commands_to_points(commands, resolution=path_resolution)
