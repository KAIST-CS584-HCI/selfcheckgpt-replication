"""
gather.py
---------
Collects all per-index JSON files in a directory into a single results.json,
sorted by wiki_bio_test_idx.

Run:
    python3 -m replication.prompt.gather
    python3 -m replication.prompt.gather --target_dir /path/to/dir
"""
import argparse
import glob
import json
import os

_parser = argparse.ArgumentParser(description="Gather per-index JSON files into results.json.")
_parser.add_argument(
    "--target_dir",
    type=str,
    default=os.path.dirname(__file__),
    help="directory containing per-index JSON files (default: directory of this script)",
)
_parser.add_argument(
    "--sort_by",
    type=str,
    default=None,
    help="field name to sort records by (default: no sorting)",
)


def main() -> None:
    args = _parser.parse_args()
    target_dir  = args.target_dir
    output_path = os.path.join(target_dir, "results.json")

    pattern = os.path.join(target_dir, "*.json")
    paths   = [p for p in glob.glob(pattern) if os.path.abspath(p) != os.path.abspath(output_path)]

    records: list[dict] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)

    if args.sort_by:
        records.sort(key=lambda r: r[args.sort_by])

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Collected {len(records)} records from {len(paths)} files → {output_path}")


if __name__ == "__main__":
    main()
