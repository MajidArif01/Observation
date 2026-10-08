"""Map global mouse scroll coordinates onto a screenshot image.

Thin CLI over the shared mapper: monitor geometry comes from
src/screenGeometry/ and the global -> local conversion from src/coords/.
The pipeline does the same work in the `mouseOverlay` stage; this script is for
one-off inspection.

Usage (single point):
    python scroll.py point --image path/to/screenshot.png --x 1181 --y 783

Usage (batch from timeline):
    python scroll.py episode/gourmet-tree

Usage (Python API):
    from scroll import mapScrollToImage
    result = mapScrollToImage("screenshot.png", x=1181, y=783)
    print(result["local_x"], result["local_y"])
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coords.coords import mapPoint, parseScrollDetail          # noqa: E402
from mouseOverlay.mouseOverlay import TimelineMouseOverlay, drawScrollMarker   # noqa: E402
from screenGeometry.screenGeometry import resolve              # noqa: E402


def mapScrollToImage(
    imagePath,
    *,
    x,
    y,
    screen=1,
    monitors=None,
    detail=None,
    draw=False,
    outputPath=None,
):
    """Map one global scroll point onto one image.

    With no `monitors`, the image is treated as the whole monitor at the origin
    implied by `screen` -- i.e. the identity map. Pass monitors from
    screenGeometry.resolve() for a multi-monitor session.
    """
    imagePath = Path(imagePath)
    img = cv2.imread(str(imagePath))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {imagePath}")

    height, width = img.shape[:2]
    if monitors is None:
        monitors = {screen: {"origin_x": 0, "origin_y": 0, "width": width,
                             "height": height, "has_screenshots": True,
                             "source": "single_image", "confidence": "low"}}

    result = mapPoint(x, y, monitors, screen, (width, height))
    dx, dy = parseScrollDetail(detail)
    result.update(image=str(imagePath), scroll_dx=dx, scroll_dy=dy)

    if draw and "draw_x" in result:
        out = Path(outputPath) if outputPath else imagePath.with_name(
            f"{imagePath.stem}_scroll{imagePath.suffix}"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), drawScrollMarker(
            img, result["draw_x"], result["draw_y"], detail
        ))
        result["output_path"] = str(out)

    return result


def mapTimelineScrolls(episodeFolder, timelinePath=None, *, draw=True):
    """Map every scroll event in an episode's timeline onto its screenshot."""
    overlay = TimelineMouseOverlay(episodeFolder, timelinePath, types=("scroll",))
    if draw:
        return overlay.run()

    episode = Path(episodeFolder).resolve()
    timeline = json.loads(
        Path(timelinePath or episode / "processdata" / "timeline.json").read_text(encoding="utf-8")
    )
    geometry = resolve(episode, timeline)
    items = []
    for item in timeline.get("items", []):
        if item.get("type") != "scroll" or item.get("source") != "mouse":
            continue
        for event in item.get("events", []):
            dx, dy = parseScrollDetail(event.get("detail"))
            items.append({
                "timeline_index": item.get("index"),
                "value": item.get("value"),
                "time": item.get("time"),
                "detail": event.get("detail"),
                "scroll_dx": dx,
                "scroll_dy": dy,
                **mapPoint(event["x"], event["y"], geometry["monitors"],
                           event.get("reported_screen", event.get("screen"))),
            })
    return {
        "count": len(items),
        "mapped": sum(i["status"] == "ok" for i in items),
        "geometry": geometry["monitors"],
        "geometry_coverage": geometry["coverage"],
        "items": items,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Map scroll coordinates onto screenshot images")
    sub = parser.add_subparsers(dest="mode")

    p1 = sub.add_parser("point", help="Map one scroll point onto one image")
    p1.add_argument("--image", "-i", required=True, help="Path to screenshot PNG")
    p1.add_argument("--x", type=float, required=True, help="Global scroll X")
    p1.add_argument("--y", type=float, required=True, help="Global scroll Y")
    p1.add_argument("--screen", type=int, default=1, help="Monitor number (default: 1)")
    p1.add_argument("--detail", default=None, help='Scroll detail, e.g. "dx=0,dy=-1"')
    p1.add_argument("--draw", action="store_true", default=True, help="Draw marker (default: on)")
    p1.add_argument("--no-draw", dest="draw", action="store_false", help="Skip drawing marker")
    p1.add_argument("--output", "-o", help="Output image path when drawing")

    p2 = sub.add_parser("timeline", help="Map all scroll events from timeline.json")
    p2.add_argument("episode", help="Episode folder or timeline.json path")
    p2.add_argument("--draw", action="store_true", default=True, help="Save annotated images")
    p2.add_argument("--no-draw", dest="draw", action="store_false", help="Skip drawing markers")

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and not argv[0].startswith("-") and argv[0] not in ("point", "timeline"):
        argv = ["timeline", *argv]        # bare `python scroll.py <episode>`
    if not argv:
        parser.print_help()
        raise SystemExit(2)

    args = parser.parse_args(argv)

    if args.mode == "point":
        result = mapScrollToImage(
            args.image, x=args.x, y=args.y, screen=args.screen,
            detail=args.detail, draw=args.draw, outputPath=args.output,
        )
        print(json.dumps(result, indent=2))
        return result

    path = Path(args.episode).resolve()
    if path.is_file() and path.name == "timeline.json":
        episode, timeline = path.parent.parent, path
    else:
        episode, timeline = path, path / "processdata" / "timeline.json"

    result = mapTimelineScrolls(episode, timeline, draw=args.draw)
    manifest = episode / "processdata" / "scroll_mapping.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Mapped {result['mapped']}/{result['count']} scroll events")
    print(f"Manifest -> {manifest}")
    return result


if __name__ == "__main__":
    main()
