import argparse
import subprocess
import time
from datetime import datetime

from weight_splice_lab.harness.budget import RateReading, latest_reading
from weight_splice_lab.harness.codex import codex_home


def describe(r: RateReading) -> str:
    reset = datetime.fromtimestamp(r.resets_at).strftime("%a %d %b %H:%M") if r.resets_at else "?"
    return f"weekly codex usage {r.used_percent:.0f}% (resets {reset})"


def notify(msg: str) -> None:
    subprocess.run(
        [
            "osascript",
            "-e",
            f'display notification "{msg}" with title "Codex usage" sound name "Glass"',
        ],
        check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--alert-percent", type=float, default=50.0)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    a = ap.parse_args()
    sessions = codex_home(a.profile) / "sessions"
    last: float | None = None
    alerted = False
    while True:
        r = latest_reading(sessions)
        if r is not None and r.used_percent != last:
            print(f"{time.strftime('%H:%M:%S')} {describe(r)}", flush=True)
            last = r.used_percent
        if r is not None and r.used_percent >= a.alert_percent and not alerted:
            msg = f"ALERT {describe(r)} is at or above {a.alert_percent:.0f}%"
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
            notify(msg)
            alerted = True
        if not a.watch:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
