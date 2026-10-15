"""Independent checker for task e): copper-wire connections (TIØ4130, 2026).

Source: "Mini project 2026 - A communication problem", pages 3-4 and 7.

Input: a pandas DataFrame OR a UTF-8 CSV/TSV with these three columns:
    City, Switch, Capacity
A row represents one installed copper cable, capacity in TB/s. Zero-capacity
connections may be omitted. No Gurobi installation or optimisation-model code
is needed. Numeric and geographic input data are independently transcribed
from the project statement (Appendix 2).

By default, read copper_wire_sol.csv and the task-c main line in main_line_sol.csv
from the project root (the parent of modules/), regardless of working directory.
Explicit file paths are resolved relative to the working directory as usual.

The checker verifies FEASIBILITY and recalculates total installation cost. It
does not prove that the submitted solution is globally cost-minimal.

Notebook example:
    from modules.check_solution_of_copper_wire import check_solution
    report = check_solution()
    report = check_solution(solution_df, reported_cost=model.ObjVal)

Terminal example:
    python modules/check_solution_of_copper_wire.py
    python modules/check_solution_of_copper_wire.py copper_wire_sol.csv --main-line main_line_sol.csv
"""

import argparse
import csv
import math
from pathlib import Path
import unicodedata


# Appendix 2: city -> (x, y, demand in TB/s). None = switching station.
CITIES = {
    "Warszawa": (100, 65, None),
    "Skierniewice": (92, 62, 197),
    "Łódź": (85, 58, 317),
    "Konin": (68, 64, 236),
    "Sieradz": (72, 55, 178),
    "Piotrków": (89, 51, 139),
    "Radom": (102, 50, 288),
    "Kielce": (98, 42, None),
    "Częstochowa": (79, 40, 278),
    "Bytom": (76, 31, 163),
    "Gliwice": (73, 30, 105),
    "Sosnowiec": (80, 29, 143),
    "Kraków": (89, 25, 331),
    "Bielsko": (79, 20, 276),
    "Katowice": (78, 28, 263),
    "Jastrzębie-Zdrój": (74, 23, 137),
    "Wodzisław Śląski": (71, 23, 119),
    "Poznań": (54, 68, 251),
    "Zielona Góra": (39, 60, 121),
    "Leszno": (51, 58, 84),
    "Kalisz": (65, 58, 207),
    "Legnica": (45, 47, 176),
    "Wrocław": (54, 45, 133),
    "Jelenia Góra": (40, 42, 167),
    "Wałbrzych": (46, 39, 129),
    "Opole": (67, 37, None),
}

STATIONS = {"Warszawa": 800.0, "Kielce": 1200.0, "Opole": 400.0}
PRICE = 1000.0                 # EUR per coordinate unit per TB/s
MAX_CABLE_CAPACITY = 200.0     # TB/s
LODZ_MAX_FRACTION = 0.4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOLUTION_PATH = PROJECT_ROOT / "copper_wire_sol.csv"
DEFAULT_MAIN_LINE_PATH = PROJECT_ROOT / "main_line_sol.csv"


def normalize(name):
    text = str(name).strip().casefold().replace("ł", "l")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if char.isalnum() and not unicodedata.combining(char))


CANONICAL_NAMES = {normalize(city): city for city in CITIES}
CANONICAL_NAMES.update({"warsaw": "Warszawa", "warszaw": "Warszawa"})


def canonical_city(value):
    name = CANONICAL_NAMES.get(normalize(value))
    if name is None:
        raise ValueError(f"Unknown city/station name: {value!r}.")
    return name


def read_csv_records(filename, required_columns):
    """Read CSV, semicolon CSV or TSV, ignoring optional exported index columns."""
    with Path(filename).open(encoding="utf-8-sig", newline="") as stream:
        first_line = stream.readline()
        stream.seek(0)
        delim = "\t" if "\t" in first_line else ";" if ";" in first_line else ","
        reader = csv.DictReader(stream, delimiter=delim, strict=True)
        if reader.fieldnames is None:
            raise ValueError("Input file is empty.")
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate column headers.")
        missing = set(required_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
        records = []
        for row in reader:
            if all(value is not None and not str(value).strip() for value in row.values()):
                continue
            if None in row or any(row.get(col) is None for col in required_columns):
                raise ValueError(f"Line {reader.line_num}: malformed row.")
            records.append(row)
    return records


def source_records(source, columns):
    """Accept a pandas DataFrame, or a path to a CSV/TSV file."""
    if hasattr(source, "columns") and hasattr(source, "to_dict"):
        missing = set(columns) - set(source.columns)
        if missing:
            raise ValueError(f"Missing DataFrame columns: {', '.join(sorted(missing))}.")
        return source.to_dict(orient="records")
    if isinstance(source, (str, Path)):
        return read_csv_records(source, columns)
    raise TypeError("Expected a pandas DataFrame or a CSV/TSV file path.")


def resolve_main_line(main_line):
    if main_line is None:
        main_line = DEFAULT_MAIN_LINE_PATH
    if isinstance(main_line, (str, Path)) or hasattr(main_line, "columns"):
        names = [r["City"] for r in source_records(main_line, ("City",))]
    else:
        names = list(main_line)
    names = [canonical_city(name) for name in names]
    if len(names) != len(set(names)):
        raise ValueError("The supplied main line contains duplicate cities.")
    missing_stations = set(STATIONS) - set(names)
    if missing_stations:
        raise ValueError(f"Main line is missing switching stations: {sorted(missing_stations)}.")
    return set(names)


def distance(city, station):
    x1, y1, _ = CITIES[city]
    x2, y2, _ = CITIES[station]
    return math.hypot(x1 - x2, y1 - y2)


def distance_limit(city, station):
    d = distance(city, station)
    return max(0.0, 2 * math.pi * min(35.0, 70.0 - d))


def parse_capacity(raw):
    try:
        value = float(str(raw).strip().replace(",", "."))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid capacity: {raw!r}.") from None
    if not math.isfinite(value):
        raise ValueError(f"Capacity must be finite: {raw!r}.")
    return value


def check_solution(solution=DEFAULT_SOLUTION_PATH, main_line=None, tolerance=1e-4,
                   reported_cost=None, cost_tolerance=0.05, verbose=True):
    """Independently check task-e feasibility and calculate installation cost.

    Parameters:
        solution: DataFrame or CSV path with City, Switch, Capacity;
            defaults to copper_wire_sol.csv in the project root.
        main_line: optional iterable of names, CSV path with City, or DataFrame;
            defaults to main_line_sol.csv in the project root.
        tolerance: absolute feasibility tolerance, TB/s.
        reported_cost: optional Gurobi objective (EUR) to cross-check.
        cost_tolerance: absolute tolerance when comparing EUR totals.
        verbose: print the individual PASS/FAIL checks and results.

    Returns a dictionary with: valid, checks, total_cost, station_usage,
    city_supply, selected_links and example_city_connections.
    """
    result = {
        "valid": False,
        "checks": [],
        "total_cost": None,
        "station_usage": {},
        "city_supply": {},
        "selected_links": [],
        "example_city_connections": {},
    }

    def add(title, passed, detail):
        result["checks"].append({
            "name": title, "passed": bool(passed), "details": detail,
        })

    def finish():
        result["valid"] = bool(result["checks"]) and all(c["passed"] for c in result["checks"])
        if verbose:
            print("\nCopper-wire checker: " + ("PASS" if result["valid"] else "FAIL"))
            for check in result["checks"]:
                print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['details']}")
            if result["total_cost"] is not None:
                print(f"\nRecalculated total cost: {result['total_cost']:,.2f} EUR")
            if result["station_usage"]:
                print("Switching-station utilization:")
                for station, use in result["station_usage"].items():
                    cap = STATIONS[station]
                    print(f"  {station}: {use:.6f} / {cap:.0f} TB/s ({100 * use / cap:.2f}%)")
            for city, connections in result["example_city_connections"].items():
                print(f"{city} -> stations (TB/s): " + ", ".join(
                    f"{s}: {cap:.6f}" for s, cap in connections.items()))
            if not result["valid"]:
                print("NOTE: An invalid solution's cost/capacity summaries are only diagnostic.")
        return result

    if not math.isfinite(tolerance) or tolerance < 0 or not math.isfinite(cost_tolerance) or cost_tolerance < 0:
        add("Input", False, "Tolerances must be finite and nonnegative.")
        return finish()

    try:
        line = resolve_main_line(main_line)
        remaining = [city for city in CITIES if city not in line]
        raw_records = source_records(solution, ("City", "Switch", "Capacity"))
        records = [
            (canonical_city(r["City"]), canonical_city(r["Switch"]), parse_capacity(r["Capacity"]))
            for r in raw_records
        ]
        if not records:
            raise ValueError("Solution has no rows.")
        if reported_cost is not None:
            reported_cost = float(reported_cost)
            if not math.isfinite(reported_cost):
                raise ValueError("Reported cost must be finite.")
    except (OSError, TypeError, ValueError, UnicodeError, csv.Error) as exc:
        add("Input", False, str(exc))
        return finish()

    add("Input", True, f"Read {len(records)} rows; expected {len(remaining)} cities outside the main line.")

    # Only links from remaining cities to the three permitted switching stations are valid.
    valid_pairs = {(city, station) for city in remaining for station in STATIONS}
    unexpected = sorted({(city, station) for city, station, _ in records
                         if (city, station) not in valid_pairs})
    add("Only permitted city-station links", not unexpected,
        f"Unexpected links: {unexpected[:8] or 'none'}.")

    pair_counts = {}
    capacities = {}
    for city, station, cap in records:
        key = (city, station)
        pair_counts[key] = pair_counts.get(key, 0) + 1
        capacities[key] = capacities.get(key, 0.0) + cap
    duplicates = sorted(key for key, count in pair_counts.items() if count > 1)
    add("No parallel cables", not duplicates,
        f"Repeated city-station pairs: {duplicates[:8] or 'none'}.")

    negative = [(city, station, cap) for city, station, cap in records if cap < -tolerance]
    add("Nonnegative cable capacities", not negative,
        f"Negative entries: {negative[:8] or 'none'}.")

    too_large = [(city, station, cap) for city, station, cap in records
                 if (city, station) in valid_pairs and cap > MAX_CABLE_CAPACITY + tolerance]
    add("200 TB/s maximum per cable", not too_large,
        f"Violations: {too_large[:8] or 'none'}.")

    distance_excess = [
        (city, station, round(cap, 6), round(distance_limit(city, station), 6))
        for city, station, cap in records
        if (city, station) in valid_pairs and cap > distance_limit(city, station) + tolerance
    ]
    add("Distance-dependent capacity", not distance_excess,
        f"Violations (city, station, actual, max): {distance_excess[:8] or 'none'}.")

    # Sparse exports are allowed: missing pairs are interpreted as 0 TB/s.
    demands = {}
    demand_violations = []
    no_links = []
    for city in remaining:
        supplied = math.fsum(capacities.get((city, station), 0.0) for station in STATIONS)
        demands[city] = supplied
        required = CITIES[city][2]
        if abs(supplied - required) > tolerance:
            demand_violations.append((city, round(supplied, 6), required))
        if not any(capacities.get((city, station), 0.0) > tolerance for station in STATIONS):
            no_links.append(city)
    result["city_supply"] = demands
    add("Demand fully met", not demand_violations,
        f"Violations (city, supplied, required): {demand_violations[:10] or 'none'}.")
    add("Every remaining city connected", not no_links,
        f"Unconnected cities: {no_links or 'none'}.")

    usage = {station: math.fsum(capacities.get((city, station), 0.0) for city in remaining)
             for station in STATIONS}
    result["station_usage"] = usage
    overloaded = [(station, round(used, 6), STATIONS[station]) for station, used in usage.items()
                  if used > STATIONS[station] + tolerance]
    add("Switching-station capacity", not overloaded,
        f"Overloaded (station, use, max): {overloaded or 'none'}.")

    if "Łódź" in remaining:
        limit = LODZ_MAX_FRACTION * CITIES["Łódź"][2]
        violations = [(station, round(capacities.get(("Łódź", station), 0.0), 6))
                      for station in STATIONS
                      if capacities.get(("Łódź", station), 0.0) > limit + tolerance]
        add("Łódź: maximum 40% per cable", not violations,
            f"Maximum {limit:.6f} TB/s per cable; violations: {violations or 'none'}.")
    else:
        add("Łódź: maximum 40% per cable", True,
            "Not applicable because Łódź is already on the main line.")

    # Independent price calculation; do NOT use the model objective or its D matrix.
    # Invalid/unexpected connections are excluded from the informative cost summary.
    calculated_cost = math.fsum(
        PRICE * distance(city, station) * cap
        for city, station, cap in records if (city, station) in valid_pairs
    )
    result["total_cost"] = calculated_cost
    if reported_cost is not None:
        difference = abs(calculated_cost - reported_cost)
        add("Reported objective cost", difference <= cost_tolerance,
            f"Calculated {calculated_cost:,.2f} EUR; reported {reported_cost:,.2f} EUR; "
            f"difference {difference:.6f} EUR (tolerance {cost_tolerance:.2f}).")
    else:
        add("Cost independently recalculated", True,
            "Cost computed as 1000 * Euclidean distance * installed TB/s; no reported cost supplied.")

    result["selected_links"] = [
        {"City": city, "Switch": station, "Capacity": cap}
        for city, station, cap in records if (city, station) in valid_pairs and cap > tolerance
    ]
    for city in ("Poznań", "Wałbrzych"):
        if city in remaining:
            result["example_city_connections"][city] = {
                station: capacities.get((city, station), 0.0) for station in STATIONS
            }
    return finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", nargs="?", default=DEFAULT_SOLUTION_PATH,
                        help="CSV with City, Switch, Capacity (default: project-root copper_wire_sol.csv).")
    parser.add_argument("--main-line", default=DEFAULT_MAIN_LINE_PATH,
                        help="Main-line CSV with column City (default: project-root main_line_sol.csv).")
    parser.add_argument("--reported-cost", type=float, help="Optional objective value to verify (EUR).")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Feasibility tolerance in TB/s, default 0.0001.")
    parser.add_argument("--cost-tolerance", type=float, default=0.05,
                        help="Cost-comparison tolerance in EUR, default 0.05.")
    args = parser.parse_args()
    report = check_solution(
        args.solution, main_line=args.main_line, tolerance=args.tolerance,
        reported_cost=args.reported_cost, cost_tolerance=args.cost_tolerance,
    )
    raise SystemExit(0 if report["valid"] else 1)
