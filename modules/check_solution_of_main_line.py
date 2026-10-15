"""Independent checker for the main-line part of the 2026 mini project.

Source: 'Mini project 2026 - A communication problem', pp. 1-3 and 7.
Assumptions: at least 4/5/5 cities in areas 1/2/3; one continuous simple
ordered path; Euclidean distances in coordinate units, ignoring terrain.
The three length columns contain whole-route totals, recorded once or
repeated consistently. Blank copies are allowed. Fixed links count towards
total length, but not within-area length.

The modelling objective is to minimize total main-line length.
Checks the exported route, not optimality or model edges absent from the file.
Bidirectional communication does not require two copies of a connection.
Uses only the Python standard library; no Gurobi installation is needed.
Defaults to main_line_sol.csv in the project root, regardless of working directory.
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import unicodedata

DEFAULT_SOLUTION_PATH = Path(__file__).resolve().parent.parent / "main_line_sol.csv"


# Appendix 2: (area, x, y).
CITIES = {
    "Warszawa": (1, 100, 65),
    "Skierniewice": (1, 92, 62),
    "Łódź": (1, 85, 58),
    "Konin": (1, 68, 64),
    "Sieradz": (1, 72, 55),
    "Piotrków": (1, 89, 51),
    "Radom": (1, 102, 50),
    "Kielce": (2, 98, 42),
    "Częstochowa": (2, 79, 40),
    "Bytom": (2, 76, 31),
    "Gliwice": (2, 73, 30),
    "Sosnowiec": (2, 80, 29),
    "Kraków": (2, 89, 25),
    "Bielsko": (2, 79, 20),
    "Katowice": (2, 78, 28),
    "Jastrzębie-Zdrój": (2, 74, 23),
    "Wodzisław Śląski": (2, 71, 23),
    "Poznań": (3, 54, 68),
    "Zielona Góra": (3, 39, 60),
    "Leszno": (3, 51, 58),
    "Kalisz": (3, 65, 58),
    "Legnica": (3, 45, 47),
    "Wrocław": (3, 54, 45),
    "Jelenia Góra": (3, 40, 42),
    "Wałbrzych": (3, 46, 39),
    "Opole": (3, 67, 37),
}

MIN_CITIES = {1: 4, 2: 5, 3: 5}
ENDPOINTS = {
    1: ("Warszawa", "Radom"),
    2: ("Kielce", "Gliwice"),
    3: ("Opole", "Jelenia Góra"),
}
FIXED_LINKS = [("Radom", "Kielce"), ("Gliwice", "Opole")]
LENGTH_COLUMNS = (
    "Within-area cable length",
    "Fixed connections length",
    "Total cable length",
)


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold().replace("ł", "l"))
    return "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))


CITY_NAMES = {normalize(city): city for city in CITIES}
CITY_NAMES.update({
    "warsaw": "Warszawa",
    "warszaw": "Warszawa",
    "piotrkowtrybunalski": "Piotrków",
    "bielskobiala": "Bielsko",
})


def distance(city_a, city_b):
    _, x_a, y_a = CITIES[city_a]
    _, x_b, y_b = CITIES[city_b]
    return math.hypot(x_a - x_b, y_a - y_b)


def read_solution(file_path):
    """Read a UTF-8 TSV, CSV or semicolon-separated solution file."""
    with Path(file_path).open(encoding="utf-8-sig", newline="") as stream:
        first_line = stream.readline()
        stream.seek(0)
        delimiter = "\t" if "\t" in first_line else ";" if ";" in first_line else ","
        reader = csv.DictReader(stream, delimiter=delimiter, strict=True)
        if reader.fieldnames is None:
            raise ValueError("The file is empty.")

        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Column names must be unique.")
        required = ("Order", "City") + LENGTH_COLUMNS
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))

        rows = []
        for row in reader:
            if all(value is not None and not str(value).strip() for value in row.values()):
                continue
            label = f"File line {reader.line_num}"
            if None in row or any(row[name] is None for name in required):
                raise ValueError(f"{label}: wrong number of fields.")
            try:
                order = int(row["Order"].strip())
            except ValueError:
                raise ValueError(f"{label}: Order must be an integer.") from None
            city = CITY_NAMES.get(normalize(row["City"]))
            if city is None:
                raise ValueError(f"{label}: unknown city {row['City']!r}.")

            lengths = {}
            for column in LENGTH_COLUMNS:
                value = row[column].strip()
                if not value:
                    lengths[column] = None
                    continue
                try:
                    number = float(value.replace(",", "."))
                except ValueError:
                    raise ValueError(f"{label}: {column} must be numeric or blank.") from None
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"{label}: {column} must be finite and nonnegative.")
                lengths[column] = number
            rows.append({"order": order, "city": city, "lengths": lengths})

    if not rows:
        raise ValueError("The file contains no cities.")
    return sorted(rows, key=lambda row: row["order"])


def check_solution(file_path=DEFAULT_SOLUTION_PATH, tolerance=0.01):
    """Return valid, individual checks, route and recalculated length totals."""
    result = {"valid": False, "checks": [], "route": [], "lengths": {}}

    def print_report():
        print("\nSolution check: " + ("PASS" if result["valid"] else "FAIL"))
        for check in result["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"[{status:4}] {check['name']}: {check['details']}")
        if result["lengths"]:
            print("\nRecalculated lengths:")
            for name, length in result["lengths"].items():
                print(f"  {name}: {length:.6f}")

    def add(name, passed, details):
        result["checks"].append({"name": name, "passed": bool(passed), "details": details})

    if not math.isfinite(tolerance) or tolerance < 0:
        add("Input", False, "Tolerance must be finite and nonnegative.")
        print_report()
        return result
    try:
        rows = read_solution(file_path)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        add("Input", False, str(error))
        print_report()
        return result
    add("Input", True, f"Read {len(rows)} city rows.")

    route = [row["city"] for row in rows]
    result["route"] = route
    selected = set(route)
    orders = [row["order"] for row in rows]
    add("Order", orders == list(range(1, len(rows) + 1)),
        "Order must contain each integer from 1 to the number of cities exactly once.")
    repeated_cities = [city for city, count in Counter(route).items() if count > 1]
    add("Simple path", not repeated_cities,
        "No repeated cities." if not repeated_cities else "Repeated cities: " + ", ".join(repeated_cities))
    add("Global endpoints", route[0] == "Warszawa" and route[-1] == "Jelenia Góra",
        f"Start: {route[0]}; end: {route[-1]}.")

    areas = [CITIES[city][0] for city in route]
    add("Area order", areas == sorted(areas),
        "The route must visit areas 1, 2 and 3 as consecutive blocks in that order.")
    for area, minimum in MIN_CITIES.items():
        segment = [city for city in route if CITIES[city][0] == area]
        count = len(set(segment))
        add(f"Area {area} city count", count >= minimum,
            f"{count} distinct cities; at least {minimum} required, including the endpoints.")
        start, end = ENDPOINTS[area]
        add(f"Area {area} endpoints", bool(segment) and segment[0] == start and segment[-1] == end,
            f"Required: {start} to {end}.")

    edges = list(zip(route, route[1:]))
    undirected_edges = [tuple(sorted(edge)) for edge in edges]
    repeated_edges = [edge for edge, count in Counter(undirected_edges).items() if count > 1]
    add("Connections used at most once", not repeated_edges,
        "Each city pair may appear at most once, counting both directions as the same connection.")
    for city_a, city_b in FIXED_LINKS:
        add(f"Fixed link {city_a} - {city_b}", edges.count((city_a, city_b)) == 1,
            "These cities must be consecutive in the required direction of route traversal.")
    cross_area_edges = [edge for edge in edges if CITIES[edge[0]][0] != CITIES[edge[1]][0]]
    add("Cross-area connections", cross_area_edges == FIXED_LINKS,
        "The only area transitions must be Radom - Kielce and Gliwice - Opole, in that order.")

    group = selected & {"Konin", "Kalisz"}
    add("Konin or Kalisz", len(group) >= 1,
        "At least one required; selected: " + (", ".join(sorted(group)) or "none") + ".")
    group = selected & {"Bytom", "Sosnowiec", "Katowice"}
    add("Bytom / Sosnowiec / Katowice", len(group) <= 1,
        "At most one allowed; selected: " + (", ".join(sorted(group)) or "none") + ".")
    group = selected & {"Wodzisław Śląski", "Jastrzębie-Zdrój"}
    add("Wodzisław Śląski / Jastrzębie-Zdrój", len(group) <= 1,
        "At most one allowed; selecting neither is allowed.")
    group = selected & {"Poznań", "Zielona Góra", "Leszno"}
    add("Poznań / Zielona Góra / Leszno", len(group) >= 1,
        "At least one required; selected: " + (", ".join(sorted(group)) or "none") + ".")

    within = math.fsum(distance(a, b) for a, b in edges if CITIES[a][0] == CITIES[b][0])
    fixed = math.fsum(distance(a, b) for a, b in edges if (a, b) in FIXED_LINKS or (b, a) in FIXED_LINKS)
    total = math.fsum(distance(a, b) for a, b in edges)
    result["lengths"] = dict(zip(LENGTH_COLUMNS, (within, fixed, total)))

    for column, expected in result["lengths"].items():
        values = [row["lengths"][column] for row in rows if row["lengths"][column] is not None]
        present = bool(values)
        consistent = present and max(values) - min(values) <= tolerance + 1e-9
        accurate = present and all(abs(value - expected) <= tolerance + 1e-9 for value in values)
        add(column, consistent and accurate,
            f"Recalculated: {expected:.6f}; tolerance: {tolerance:g}. "
            + ("All recorded copies must agree and match the recalculated total." if present
               else "At least one recorded value is required."))

    result["valid"] = all(check["passed"] for check in result["checks"])
    print_report()
    return result
