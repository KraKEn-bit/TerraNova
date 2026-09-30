"""Write Earth's observed envelope and quantiles into ``data/normalization.json``.

    python -m src.acquire.calibrate

For every criterion, over candidate land cells:

* ``earth_min`` / ``earth_max`` - the 0.5th and 99.5th percentiles. A target
  outside this envelope is scored against the nearest edge (see
  ``CriterionRange.effective_target``). Percentiles rather than the absolute
  extremes keep one odd cell from defining "Earth's most extreme value".
* ``earth_quantiles`` - the 0th, 5th, ..., 100th percentiles. Targets whose
  measured value is not comparable at our 0.5 degree scale (slope, roughness)
  are specified as an Earth percentile and resolved through this table.

Scoring ranges (``min`` / ``max``) and weights are editorial choices and are
left untouched. Run this after ``build_predictor_stack``; it reads
``cache/derived``.
"""

from __future__ import annotations

import json

import numpy as np

from src.acquire.build_predictor_stack import DERIVED_DIR, LAND_FRACTION_MIN, NORM_PATH
from src.compute.similarity import CRITERIA, QUANTILE_LEVELS


def main() -> int:
    land = np.load(DERIVED_DIR / "land_fraction.npy") >= LAND_FRACTION_MIN
    arrays = {key: np.load(DERIVED_DIR / f"{key}.npy") for key in CRITERIA}
    scorable = land.copy()
    for values in arrays.values():
        scorable &= np.isfinite(values)

    norm = json.loads(NORM_PATH.read_text(encoding="utf-8"))
    for key in CRITERIA:
        values = arrays[key][scorable].astype(np.float64)
        low, high = np.percentile(values, [0.5, 99.5])
        entry = norm["criteria"][key]
        entry["earth_min"] = round(float(low), 4)
        entry["earth_max"] = round(float(high), 4)
        entry["earth_quantiles"] = [
            round(float(q), 4) for q in np.percentile(values, QUANTILE_LEVELS)
        ]
        print(f"[calibrate] {key:26s} earth envelope {low:10.3f} .. {high:10.3f}")
    norm["calibration"] = {
        "scorable_cells": int(scorable.sum()),
        "envelope": "0.5th to 99.5th percentile over scorable land cells",
        "quantile_levels": list(QUANTILE_LEVELS),
    }
    NORM_PATH.write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")
    print(f"[calibrate] updated {NORM_PATH.name} from {int(scorable.sum())} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
