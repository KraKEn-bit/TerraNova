"""Print the known-analog validation for every target.

A good model ranks known analog sites high and densely vegetated reference
points low; the headline number is ROC-AUC (see src/compute/validation.py).

    OFFLINE=1 python -m scripts.check_controls
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def main() -> None:
    client = TestClient(app)
    for target in client.get("/api/targets").json()["targets"]:
        report = client.get("/api/validation", params={"target_id": target["id"]}).json()
        print(
            f"\n== {target['id']} ({report['body']})  ROC-AUC {report['auc']}  "
            f"median known-analog percentile {report['median_positive_percentile']}"
        )
        for c in report["controls"]:
            score = "  n/a" if c["score"] is None else f"{c['score']:.3f}"
            pct = "  n/a" if c["percentile"] is None else f"{c['percentile']:5.1f}"
            print(f"  [{c['role']:8s}] {c['name'][:36]:36s} score={score}  percentile={pct}")


if __name__ == "__main__":
    main()
