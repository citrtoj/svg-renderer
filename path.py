import re
from enum import Enum
from pprint import pprint


type Point = tuple[float, float]
type PathCommand = tuple[str, tuple[float]]


def reflection(point: Point, current_point: Point):
    return (2 * current_point[0] - point[0], 2 * current_point[1] - point[1])


def cubic_bezier(point_0: Point, control_point_0: Point, control_point_1: Point, point_1: Point, n):
    """
    Generate intermediary points along a cubic Bézier curve.

    Args:
        point_0 (Point): Start point (x0, y0).
        point_1 (Point): End point (x1, y1).
        control_point_0 (Point): First control point (cx0, cy0).
        control_point_1 (Point): Second control point (cx1, cy1).
        n (int): Number of intermediary points to generate.

    Returns:
        list[Point]: List of (x, y) tuples representing points along the curve.
    """

    def bezier_interpolation(t, point_0: Point, control_point_0: Point, control_point_1: Point, point_1: Point):
        """Calculate a point on a cubic Bézier curve at parameter t."""
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

    points = [
        bezier_interpolation(t, point_0, control_point_0, control_point_1, point_1)
        for t in [i / (n - 1) for i in range(n)]
    ]

    return points


def quadratic_bezier(point_0: tuple, control_point: tuple, point_1: tuple, n: int):
    """
    Generate intermediary points along a quadratic Bézier curve.

    Args:
        point_0 (tuple): Start point (x0, y0).
        control_point (tuple): Control point (cx, cy).
        point_1 (tuple): End point (x1, y1).
        n (int): Number of intermediary points to generate.

    Returns:
        list[tuple]: List of (x, y) tuples representing points along the curve.
    """

    def bezier_interpolation(t, point_0: tuple, control_point: tuple, point_1: tuple):
        """Calculate a point on a quadratic Bézier curve at parameter t."""
        x = (1 - t) ** 2 * point_0[0] + 2 * (1 - t) * t * control_point[0] + t**2 * point_1[0]
        y = (1 - t) ** 2 * point_0[1] + 2 * (1 - t) * t * control_point[1] + t**2 * point_1[1]
        return (x, y)

    points = [bezier_interpolation(t, point_0, control_point, point_1) for t in [i / (n - 1) for i in range(n)]]

    return points


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

    def get_length(command: str):
        if len(command) != 1:
            raise ValueError(f"Invalid command {command}; must be 1 letter")
        command_upper = command.upper()
        if command_upper not in PathCommandTypes.__members__:
            raise ValueError(f"Invalid command {command}; not in list {PathCommandTypes.__members__.keys()}")
        return PathCommandTypes.__members__[command_upper].value


def parse_path_commands(path_data: str) -> list[PathCommand]:
    commands = []

    pattern = re.compile(r"([a-zA-Z])|([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    matches = pattern.findall(path_data)

    current_command = None
    current_numbers = []

    def add_commands(command, numbers):
        if command is not None:
            args_len = PathCommandTypes.get_length(command)

            if args_len == 0:
                if len(numbers) == 0:
                    commands.append((command, tuple()))
                else:
                    raise ValueError(f"Invalid command {command} with too many numbers ({len(numbers)})")

            else:
                if len(numbers) % args_len != 0:
                    print(args_len)
                    print(len(numbers))
                    raise ValueError(f"Not enough numbers ({len(numbers)}) for all commands of same type {command}")
                for i in range(len(numbers) // args_len):
                    commands.append(
                        (
                            command,
                            tuple(numbers[j] for j in range(i * args_len, (i + 1) * args_len)),
                        )
                    )

    for match in matches:
        if match[0]:
            add_commands(current_command, current_numbers)
            current_command = match[0]
            current_numbers = []
        elif match[1]:
            current_numbers.append(float(match[1]))

    if current_command is not None:
        add_commands(current_command, current_numbers)

    return commands


def relative_to_absolute(command: PathCommand, current_point) -> PathCommand:
    if command[0].upper() == command[0]:
        return command

    new_command_type = command[0].upper()
    new_coords = []

    for i, coord in enumerate(command[1]):
        match new_command_type:
            case "H":
                new_coords.append(current_point[0] + coord)
            case "V":
                new_coords.append(current_point[1] + coord)
            case "A":
                # raise NotImplementedError("Path segments of type A not supported yet")
                pass
            case _:
                new_coords.append(current_point[i % 2] + coord)

    return (new_command_type, tuple(new_coords))


def path_commands_to_points(commands: list[PathCommand], resolution) -> list[list[Point]]:
    current_point: Point = (0, 0)
    paths: list[list[Point]] = []
    current_points: list[Point] = []

    def close_path():
        nonlocal current_points, current_point, paths
        if len(current_points) >= 2:
            paths.append(current_points)
        current_points = [current_point]

    def command_values_to_points(values: tuple[float]):
        return [(values[i], values[i + 1]) for i in range(0, len(values)) if i % 2 == 0]

    for i, command in enumerate(commands):
        absolute_command = relative_to_absolute(command, current_point)
        values = absolute_command[1]

        match absolute_command[0]:
            case "M":
                point = (values[0], values[1])
                current_point = point
                if i > 0 and commands[i - 1][0].upper() == "M":
                    # it's a lineto! add to current points
                    current_points.append(point)
                else:
                    # first moveto in the sequence
                    close_path()

                current_points.append(current_point)
            case "Z":
                current_points.append(current_points[0])
                close_path()
                assert current_points == [current_point]
            case "C":
                command_points = command_values_to_points(values)
                if len(command_points) != 3:
                    raise ValueError("Invalid SVG path")
                bezier_points = cubic_bezier(current_point, *command_points, resolution)
                current_point = command_points[-1]
                bezier_points.append(current_point)
                current_points.extend(bezier_points)
            case "S":
                first_control_point = current_point
                prev_command = commands[i - 1]
                if i > 0 and prev_command[0].upper() in {"S", "C"}:
                    prev_absolute_command = relative_to_absolute(prev_command, current_point)
                    prev_values = command_values_to_points(prev_absolute_command[1])
                    first_control_point = reflection(
                        prev_values[1 if prev_absolute_command[0].upper() == "C" else 0], current_point
                    )
                command_points = command_values_to_points(values)
                if len(command_points) != 2:
                    raise ValueError("Invalid SVG path")
                bezier_points = cubic_bezier(current_point, first_control_point, *command_points, resolution)

                current_point = command_points[-1]
                bezier_points.append(current_point)
                current_points.extend(bezier_points)
            case "V":
                current_points.append((current_point[0], values[0]))
                current_point = current_points[-1]
            case "H":
                current_points.append((values[0], current_point[1]))
                current_point = current_points[-1]
                print(current_point)
            case "L":
                current_points.append((values[0], values[1]))
                current_point = current_points[-1]
            case "A":
                pass
                # not yet implemented; will ignore
            case _:
                pass

    return paths


def path_data_to_points(data: str, resolution=1000) -> list[list[Point]]:
    """Return list of subpaths (lists of points) from a path data string resulted from the
    `d` attribute of `<path>` tags in SVGs.

    Args:
        data (str): path data string
        resolution (int, optional): amount of points to use when drawing curves. Defaults to 1000.

    Returns:
        list[Point]: list of absolute points to draw on the image, in given order
    """
    commands = parse_path_commands(data)
    return path_commands_to_points(commands, resolution=resolution)


if __name__ == "__main__":
    # path_data = "M0.6.5 300 500"
    path_data = "m18.39 298.8c3.36-.03 6.72-.03 10.07-.02 2.32 6.84 5.09 13.52 7.23 20.43 2.33-6.9 5.09-13.63 7.61-20.45l9.72.03c.01 14.5-.02 29 .02 43.5-2.41-.16-5.65 1.33-7.49-.56-.14-10.5-.53-21.06.34-31.54-3.81 8.1-6.76 16.56-10.22 24.81-3.19-8.23-6.7-16.33-9.63-24.66-.1 10.76.02 21.53-.06 32.3-2.43-.34-5.39.68-7.48-.87-.28-14.32-.01-28.65-.11-42.97zm54.63-.04c8.67.02 17.34-.03 26.01 0v6.44c-2.99.01-5.97.01-8.95.01-.11 12.45.01 24.9-.07 37.36-2.66 0-5.31.01-7.97.03-.15-12.37-.03-24.74-.06-37.1-2.96-.02-5.92-.03-8.88-.02-.04-2.24-.06-4.48-.08-6.72zm29.04 6c3.92-7.83 16.18-9.38 21.79-2.57 2.58 2.56 3.11 6.28 3.57 9.72-2.54.03-5.09.03-7.63.01-.51-1.65-.55-3.52-1.63-4.93-1.35-1.05-2.99-1.6-4.52-2.31-2.27 1.26-5.46 2.58-5.28 5.73.99 3.51 4.71 4.93 7.57 6.61 4.87 2.52 10.58 5.6 11.87 11.43 1.47 6.27-3.24 12.68-9.28 14.21-5.1.92-11.05.69-14.96-3.16-3.29-2.85-3.76-7.4-4.39-11.42h7.71c.85 2.51 1.11 5.94 3.99 7.03 2.59 1.43 5.52.26 8.14-.46.55-2.37 1.66-5.12-.26-7.19-4.03-4.37-10.57-5.19-14.66-9.55-3.49-3.3-4.04-8.94-2.03-13.15zm30.48-5.97c2.57-.01 5.15-.02 7.73-.02.09 10.55-.14 21.12.07 31.67.21 5.17 7.57 7.02 10.98 3.65 2.02-1.62 1.85-4.4 1.9-6.71-.04-9.54-.03-19.07-.02-28.61 2.58 0 5.17.01 7.76.03.01 9.83-.03 19.66.03 29.48.02 2.72-.11 5.56-1.46 7.99-4.92 9.34-20.84 9.11-25.55-.32-1.54-2.93-1.45-6.35-1.5-9.55.07-9.2-.01-18.41.06-27.61zm35.73-.02c6.16.09 12.33-.27 18.48.15 6.56.79 11.24 8.19 8.99 14.46-.87 3.06-3.85 4.62-6.51 5.87 4.58 1.62 8.43 5.95 8.16 11 .29 3.69-1.33 7.4-4.25 9.66-2.62 2.18-6.19 2.32-9.42 2.58-5.14.15-10.28.05-15.41.09-.07-14.61-.01-29.21-.04-43.81m8.09 6.8c-.02 10.07-.01 20.14 0 30.21 3.03-.09 6.15.29 9.13-.46 4.47-1.14 5.48-7.99 1.83-10.62-2.3-1.9-5.45-1.73-8.24-2.07.01-2.12.02-4.24.04-6.35 1.85-.26 3.76-.26 5.54-.87 2.94-.9 4.31-4.85 2.53-7.37-2.22-3.8-7.25-2.07-10.83-2.47zm25.86-6.82c2.75 0 5.5.01 8.26.02-.08 14.62.08 29.25-.09 43.87-2.72-.02-5.43-.03-8.14-.03-.06-14.62-.01-29.24-.03-43.86zm16.74 4.97c4.31-6.26 14.21-7.8 19.94-2.6 3.25 2.63 4.29 6.87 4.61 10.85-2.43.04-4.86.05-7.29.09-.82-2.53-1.39-6.22-4.7-6.48-3.72-1.1-8.19 2.38-6.33 6.35 4.73 5.8 13.62 6.66 17.52 13.42 3.79 6.2-.07 14.79-6.74 17.01-5.68 1.6-12.72 1.23-16.93-3.42-2.87-2.83-3.21-7.01-3.34-10.79 2.46-.01 4.93 0 7.39 0 .75 3.1 1.68 7.2 5.55 7.6 3.97 1.36 8.97-2.07 7.33-6.48-3-5.71-10.37-6.27-14.85-10.41-4.56-3.44-5.47-10.55-2.16-15.14zm29.42-4.94 8.06-.06c0 5.76.01 11.51-.01 17.27 5.49.11 10.98.11 16.47.01-.02-5.76-.01-11.51-.01-17.27 2.51.01 5.02.02 7.53.02-.03 14.61.09 29.23-.06 43.84-2.48-.01-4.95-.01-7.42 0-.14-6.56.09-13.11-.17-19.66-.59 6.56-.23 13.15-.52 19.73-.13-6.55-.05-13.1.24-19.64-5.11.08-10.23-.19-15.33.09-.96 1.19-.63 2.82-.75 4.23 0 5.09.09 10.17-.04 15.26-2.66-.02-5.31-.03-7.97-.04-.03-14.59 0-29.19-.02-43.78zm39.6-.05c2.72.01 5.45.02 8.18.02-.02 14.62.03 29.25-.03 43.87-2.71-.01-5.41 0-8.11.01-.1-14.63-.03-29.27-.04-43.9zm-227.53.05c2.87-.03 5.73-.03 8.6-.02-.01 14.62.02 29.24-.02 43.87-2.62 0-5.25.09-7.86-.1-.92-1.17-.63-2.76-.75-4.13.06-13.21.02-26.42.03-39.62zm52.88 60.57c5.65-.76 11.71 1.73 14.95 6.49 4.11 5.95 4.72 13.6 3.97 20.61-.7 5.63-2.88 11.55-7.66 14.97-7.19 5.16-18.8 3.03-22.95-5.05-4.36-7.96-4.36-17.82-1.35-26.24 1.96-5.63 7.04-10.13 13.04-10.78m-.09 7.67c-3.62 1.1-5.44 4.83-6.21 8.26-1.37 5.87-1.2 12.46 1.81 17.81 1.69 3.15 5.59 4.52 8.97 3.57 4.26-1.99 6.29-6.85 6.38-11.33.21-5.23.35-11.01-2.8-15.48-1.75-2.59-5.17-3.84-8.15-2.83zm60.03-7.74c5.12-.5 10.58 1.52 13.81 5.61 3.99 4.84 5.07 11.36 5.08 17.46-.18 6.4-1.81 13.25-6.53 17.87-4.75 4.56-12.38 5.06-18.14 2.27-5.68-2.97-8.66-9.37-9.38-15.5-.5-6.21-.38-12.79 2.43-18.49 2.38-4.92 7.19-8.73 12.73-9.22m-.19 7.64c-4.12 1.22-6.08 5.64-6.76 9.55-.61 4.97-.78 10.27 1.23 14.96 1.21 2.86 3.79 5.58 7.12 5.48 4.03.39 7.08-3.18 8.14-6.71 1.86-6.09 1.74-12.97-.9-18.8-1.36-3.4-5.27-5.48-8.83-4.48zm56.87-.28c3.6-7.84 15.22-9.96 21.35-3.9 3.1 2.54 3.95 6.59 4.28 10.38-2.56.02-5.11.02-7.66-.01-.62-2.16-.7-4.97-3.07-6.02-2.54-1-6.02-.63-7.55 1.89-.83 1.82-.73 4.38 1.03 5.63 5.29 4.39 12.91 6 16.59 12.23 3.57 6.3-.47 14.72-7.21 16.72-5.85 1.51-13.2 1.09-17.16-4.09-2.5-2.69-2.57-6.48-2.88-9.92 2.59.03 5.18.07 7.77.18.55 1.95.49 4.32 2.16 5.76 3.45 3.32 11.52 1.52 10.55-4.22-1.78-5.68-8.6-6.49-12.89-9.62-5.2-2.79-7.94-9.56-5.31-15.01zm-171.69-6.71c3.34-.01 6.67-.01 10.01-.04 2.59 6.82 5.18 13.64 7.36 20.6 2.31-6.95 4.95-13.79 7.65-20.59 3.24.02 6.48.02 9.72.03.01 14.56.03 29.13 0 43.69h-7.66c-.09-10.65.1-21.3-.12-31.94-2.89 8.33-6.46 16.41-9.63 24.64-3.21-8.28-6.69-16.45-9.57-24.85-.07 10.71 0 21.43-.04 32.15h-7.7c-.05-14.56-.03-29.13-.02-43.69zm74.03 0c8.65-.01 17.31.01 25.96-.01v6.68c-3.01 0-6.01.01-9.02.04.03 12.33.04 24.66 0 36.99-2.67-.01-5.34-.01-8-.01-.05-12.34-.01-24.68-.02-37.02h-8.91c-.01-2.22-.01-4.45-.01-6.67z"
    pprint(parse_path_commands(path_data))

    pprint(reflection((200, 100), (200, 200)))
