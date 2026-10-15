"""Plot the complete network using the exported solutions and original city data.

Run from any directory:
    python final_visualization.py
    python final_visualization.py --no-show --output /tmp/final_network.png

The default output is final_visualization.png beside this script. No notebooks,
input files, or optimisation models are modified or executed.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


PROJECT_ROOT = Path(__file__).resolve().parent
AREA_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
COPPER_COLOR = "#8055a5"
# Existing inter-area cables, also shown as fixed connections in main_line.ipynb.
EXISTING_LINKS = (("Radom", "Kielce"), ("Gliwice", "Opole"))
STATION_CAPACITY = {"Warszawa": 800.0, "Kielce": 1200.0, "Opole": 400.0}
PRICE_PER_DISTANCE_AND_CAPACITY = 1000.0


def read_rows(path):
    """Read the tab-separated city/main-line files and comma-separated export."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        first_line = stream.readline()
        stream.seek(0)
        delimiter = "\t" if "\t" in first_line else ";" if ";" in first_line else ","
        return list(csv.DictReader(stream, delimiter=delimiter))


def load_network(data_dir, main_line_path, copper_path):
    cities = {}
    for area in AREA_COLORS:
        for row in read_rows(Path(data_dir) / f"Area_{area}.txt"):
            cities[row["City"]] = {
                "xy": (float(row["x"]), float(row["y"])),
                "area": area,
                "demand": None if row["TB/s"] == "-" else float(row["TB/s"]),
            }
    route = [row["City"] for row in sorted(
        read_rows(main_line_path), key=lambda row: int(row["Order"])
    )]
    if not route or len(route) != len(set(route)):
        raise ValueError("Main-line solution must contain a nonempty route of unique cities.")
    for city in route:
        if city not in cities:
            raise ValueError(f"Unknown main-line city: {city!r}.")
    for station in STATION_CAPACITY:
        if station not in route:
            raise ValueError(f"Main line is missing switching station {station!r}.")

    links = []
    for row in read_rows(copper_path):
        city, station, capacity = row["City"], row["Switch"], float(row["Capacity"])
        if city not in cities or station not in STATION_CAPACITY:
            raise ValueError(f"Unknown copper-wire endpoints: {city!r}, {station!r}.")
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f"Invalid capacity: {capacity!r}.")
        if city in route:
            raise ValueError(f"Copper-wire city {city!r} is already on the main line.")
        if capacity > 0:
            links.append((city, station, capacity))
    return cities, route, links


def place_capacity_labels(fig, ax, cities, links, city_labels):
    """Choose positions along each cable while avoiding city and cable labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [label.get_window_extent(renderer).expanded(1.08, 1.15)
                for label in city_labels]
    for city in cities.values():
        px, py = ax.transData.transform(city["xy"])
        occupied.append(Bbox.from_bounds(px - 6, py - 6, 12, 12))

    # Shorter cables have less room for labels, so place those first.
    ordered = sorted(links, key=lambda link: math.dist(
        cities[link[0]]["xy"], cities[link[1]]["xy"]
    ))
    for city, station, capacity in ordered:
        start = cities[city]["xy"]
        end = cities[station]["xy"]
        display_start = ax.transData.transform(start)
        display_end = ax.transData.transform(end)
        dx, dy = display_end - display_start
        length = math.hypot(dx, dy)
        normal = (-dy / length, dx / length)
        text = ax.annotate(
            f"{capacity:.1f}", start, xytext=(0, 0), fontsize=6.5, color=COPPER_COLOR,
            ha="center", va="center", textcoords="offset points", zorder=7,
            bbox={"boxstyle": "round,pad=0.13", "fc": "white", "ec": "none", "alpha": 0.94},
        )
        best = None
        for fraction in (0.48, 0.32, 0.64, 0.22, 0.76, 0.4, 0.56, 0.85):
            anchor = (start[0] + fraction * (end[0] - start[0]),
                      start[1] + fraction * (end[1] - start[1]))
            for offset in (5, -5, 10, -10, 15, -15):
                text.xy = anchor
                text.set_position((normal[0] * offset, normal[1] * offset))
                box = text.get_window_extent(renderer).expanded(1.12, 1.2)
                overlap = sum(
                    max(0, min(box.x1, other.x1) - max(box.x0, other.x0))
                    * max(0, min(box.y1, other.y1) - max(box.y0, other.y0))
                    for other in occupied
                )
                score = overlap * 100 + abs(offset) + abs(fraction - 0.48)
                if best is None or score < best[0]:
                    best = (score, anchor, text.get_position(), box)
        _, text.xy, position, box = best
        text.set_position(position)
        occupied.append(box)


def create_visualization(data_dir=PROJECT_ROOT / "data",
                         main_line_path=PROJECT_ROOT / "main_line_sol.csv",
                         copper_path=PROJECT_ROOT / "copper_wire_sol.csv"):
    """Return a Matplotlib figure and axes without displaying or saving them."""
    cities, route, links = load_network(data_dir, main_line_path, copper_path)
    selected = set(route)
    existing_pairs = {frozenset(link) for link in EXISTING_LINKS}
    fig, ax = plt.subplots(figsize=(12, 8))

    for city, station, capacity in links:
        start, end = cities[city]["xy"], cities[station]["xy"]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=COPPER_COLOR,
                linewidth=0.5 + 2.5 * capacity / 200.0, alpha=0.48, zorder=1)

    for start, end in zip(route, route[1:]):
        if frozenset((start, end)) in existing_pairs:
            continue  # Draw existing connections once, using their own style.
        if cities[start]["area"] != cities[end]["area"]:
            raise ValueError(f"Unexpected inter-area main-line link: {start} - {end}.")
        x1, y1 = cities[start]["xy"]
        x2, y2 = cities[end]["xy"]
        ax.plot([x1, x2], [y1, y2], color=AREA_COLORS[cities[start]["area"]],
                linewidth=2, zorder=2)

    for start, end in EXISTING_LINKS:
        x1, y1 = cities[start]["xy"]
        x2, y2 = cities[end]["xy"]
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="--", linewidth=2, zorder=2)

    outside = [city for name, city in cities.items() if name not in selected]
    ax.scatter([city["xy"][0] for city in outside], [city["xy"][1] for city in outside],
               color="lightgray", s=35, edgecolors="gray", linewidths=0.4, zorder=3)
    for area, color in AREA_COLORS.items():
        points = [cities[name]["xy"] for name in route if cities[name]["area"] == area]
        ax.scatter([xy[0] for xy in points], [xy[1] for xy in points],
                   color=color, s=65, zorder=4)
    for station in STATION_CAPACITY:
        ax.scatter(*cities[station]["xy"], color=AREA_COLORS[cities[station]["area"]],
                   marker="D", s=90, edgecolors="black", linewidths=1, zorder=5)

    label_offsets = {
        "Wodzisław Śląski": (-5, 5), "Bytom": (-5, 5), "Katowice": (5, -12),
        "Gliwice": (-5, -10), "Sosnowiec": (6, 2),
    }
    city_labels = []
    for name, city in cities.items():
        offset = label_offsets.get(name, (5, 5))
        city_labels.append(ax.annotate(
            name, city["xy"], xytext=offset, textcoords="offset points", fontsize=8,
            ha="right" if offset[0] < 0 else "left", zorder=8,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 0.3},
        ))

    total_length = sum(math.dist(cities[a]["xy"], cities[b]["xy"])
                       for a, b in zip(route, route[1:]))
    cost = sum(PRICE_PER_DISTANCE_AND_CAPACITY * capacity
               * math.dist(cities[city]["xy"], cities[station]["xy"])
               for city, station, capacity in links)
    supplied = sum(capacity for _, _, capacity in links)
    demand = sum(city["demand"] for name, city in cities.items() if name not in selected)
    usage = {station: sum(capacity for _, target, capacity in links if target == station)
             for station in STATION_CAPACITY}

    handles = [Line2D([], [], color=color, marker="o", linewidth=2, markersize=6,
                      label=f"Area {area}: main line") for area, color in AREA_COLORS.items()]
    handles.extend([
        Line2D([], [], color="black", linestyle="--", linewidth=2, label="Existing connections"),
        Line2D([], [], color=COPPER_COLOR, linewidth=2, alpha=0.65, label="Copper wires"),
        Line2D([], [], color="gray", marker="o", linestyle="none", markerfacecolor="lightgray",
               markersize=5, label="Cities served by copper"),
        Line2D([], [], color="black", marker="D", linestyle="none", markerfacecolor="white",
               markersize=6, label="Switching stations"),
    ])
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.95)
    summary = [
        f"Main line: {len(route)} cities | length: {total_length:.2f}",
        f"Copper: {len(links)} cables | {len(outside)} cities",
        f"Copper supply / demand: {supplied:,.1f} / {demand:,.1f} TB/s",
        f"Copper installation: EUR {cost:,.2f}",
        "Switching stations (used / available TB/s):",
    ]
    summary.extend(f"  {station}: {usage[station]:,.1f} / {limit:,.0f}"
                   for station, limit in STATION_CAPACITY.items())
    ax.text(0.985, 0.025, "\n".join(summary), transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", zorder=10,
            bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "lightgray", "alpha": 0.95})
    ax.set_title("Complete communication network - main line and copper wires")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.15)
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.015,
             "Copper labels: installed capacity in TB/s (rounded to 0.1). Line width scales with capacity.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    place_capacity_labels(fig, ax, cities, links, city_labels)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--main-line", type=Path, default=PROJECT_ROOT / "main_line_sol.csv")
    parser.add_argument("--copper-wire", type=Path, default=PROJECT_ROOT / "copper_wire_sol.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "final_visualization.png")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    try:
        fig, _ = create_visualization(args.data_dir, args.main_line, args.copper_wire)
        fig.savefig(args.output, dpi=220)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Could not create network map: {error}\n")
    print(f"Saved network map: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
