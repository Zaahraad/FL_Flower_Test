import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


BASE_URL = (
    "https://media.githubusercontent.com/media/"
    "buds-lab/building-data-genome-project-2/master/data"
)
RAW_FILES = {
    "electricity_cleaned.csv": "meters/cleaned/electricity_cleaned.csv",
    "metadata.csv": "metadata/metadata.csv",
    "weather.csv": "weather/weather.csv",
}
RAW_DIR = Path("data/raw")

# Quality rules over the full 2016-2017 period (17,544 hours).
MAX_MISSING = 0.001  # at most 0.1% missing hours (~17 h)
MAX_GAP_HOURS = 3  # longest missing run; these get interpolated
MAX_ZERO = 0.001  # at most 0.1% zero readings
MAX_FLAT_HOURS = 6  # longest run of identical readings (stuck meter)
MIN_MEAN_KWH = 5.0  # drop tiny meters whose signal is mostly noise


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, remote in RAW_FILES.items():
        path = RAW_DIR / name
        if path.exists():
            print(f"Already downloaded: {path}")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{remote}", path)


def longest_run(mask: np.ndarray):
    """Length of the longest run of True values."""
    if not mask.any():
        return 0
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return int((ends - starts).max())


def quality_table(meters: pd.DataFrame):
    rows = []
    for building, s in meters.items():
        v = s.to_numpy()
        repeated = np.r_[False, v[1:] == v[:-1]]
        rows.append({
            "building_id": building,
            "missing": s.isna().mean(),
            "longest_gap": longest_run(s.isna().to_numpy()),
            "zero": (s == 0).mean(),
            # +1 because a run of k repeats spans k+1 identical readings
            "longest_flat": longest_run(repeated) + 1,
            "mean_kwh": s.mean(),
        })
    return pd.DataFrame(rows).set_index("building_id")


def select_buildings(quality: pd.DataFrame, n: int) -> list[str]:
    good = quality[
        (quality.missing <= MAX_MISSING)
        & (quality.longest_gap <= MAX_GAP_HOURS)
        & (quality.zero <= MAX_ZERO)
        & (quality.longest_flat <= MAX_FLAT_HOURS)
        & (quality.mean_kwh >= MIN_MEAN_KWH)
    ].copy()
    print(f"{len(good)} of {len(quality)} buildings pass the quality rules")
    if len(good) < n:
        raise ValueError(f"Only {len(good)} buildings qualify, need {n}")

    # Building ids look like <site>_<use>_<name>, e.g. Fox_office_Joy.
    good["site"] = good.index.str.split("_").str[0]

    # Within each site: most complete first, name as a deterministic tie-break.
    queues = {
        site: list(g.sort_values(["missing", "longest_flat"]).index)
        for site, g in good.sort_index().groupby("site")
    }
    # Take one building per site in turn until we have n.
    selected = []
    while len(selected) < n:
        for site in sorted(queues):
            if queues[site] and len(selected) < n:
                selected.append(queues[site].pop(0))
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-buildings", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("Data"))
    args = parser.parse_args()

    download()

    meters = pd.read_csv(
        RAW_DIR / "electricity_cleaned.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    ).asfreq("h")
    metadata = pd.read_csv(RAW_DIR / "metadata.csv", index_col="building_id")

    quality = quality_table(meters)
    selected = select_buildings(quality, args.num_buildings)

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.csv"):
        old.unlink()  # load_data uses every CSV here, so start clean

    for building in selected:
        # Gaps are at most MAX_GAP_HOURS long, so linear interpolation is safe.
        s = meters[building].interpolate(limit_direction="both")
        pd.DataFrame({
            "datetime": s.index.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption": s.round(4).to_numpy(),
        }).to_csv(args.out / f"{building}.csv", index=False)

    # Manifest: which building became which Flower client (partition-id).
    manifest = quality.loc[selected].join(
        metadata[["site_id", "primaryspaceusage", "sqm", "timezone"]]
    )
    manifest.insert(0, "partition_id", range(len(selected)))
    manifest.to_csv(RAW_DIR.parent / "selected_buildings.csv")

    print(f"\nWrote {len(selected)} buildings to {args.out}/")
    print(manifest[["partition_id", "site_id", "primaryspaceusage", "mean_kwh"]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
