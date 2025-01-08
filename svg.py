import argparse
import xml.etree.ElementTree
import sys

from renderer import Renderer


def parse_xml(xml_path: str):
    return xml.etree.ElementTree.parse(xml_path)


def get_args():
    parser = argparse.ArgumentParser(description="Render an SVG file to a PNG image.")
    parser.add_argument("svg_path", type=str, help="path to SVG path")
    parser.add_argument(
        "-o", "--output", type=str, default="output.png", help="output PNG path (default is 'output.png')"
    )

    args = parser.parse_args()
    return args


def main():
    try:
        args = get_args()
        xml = parse_xml(args.svg_path)
        renderer = Renderer().render(xml)
        image = renderer.image
        image.save(args.output)
        print(f'\033[92mSuccessfully saved output image to: "{args.output}"\033[0m')
    except Exception as e:
        exception_name = type(e).__name__
        print(f"\033[91mError while processing image ({exception_name}): \033[0m{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
