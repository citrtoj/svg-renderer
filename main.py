import argparse
import xml.etree.ElementTree

from renderer import Renderer


def parse_xml(xml_path: str):
    return xml.etree.ElementTree.parse(xml_path)


def get_args():
    parser = argparse.ArgumentParser(description="Render an SVG file to a PNG image.")
    parser.add_argument("svg_path", type=str, help="path to SVG path.")
    parser.add_argument(
        "-o", "--output", type=str, default="output.png", help="The output PNG path (default is 'output.png')."
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    xml = parse_xml(args.svg_path)
    renderer = Renderer().render(xml)
    image = renderer.image
    image.save("output.png")


if __name__ == "__main__":
    main()
