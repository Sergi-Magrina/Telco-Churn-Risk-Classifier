from __future__ import annotations

from src.config import DRIFT_REPORT_PATH
from src.data.load_data import load_telco_data
from src.data.split import train_test_split_leakage_safe
from src.monitoring.drift import compute_drift_report
from src.utils.io import save_json
from src.utils.seed import set_global_seed


def main() -> None:
    set_global_seed()

    X, y = load_telco_data()
    X_train, X_test, y_train, y_test = train_test_split_leakage_safe(X, y)

    report = compute_drift_report(X_train, X_test)
    save_json(report, DRIFT_REPORT_PATH)
    print(f"Drift report written to {DRIFT_REPORT_PATH}")


if __name__ == "__main__":
    main()

