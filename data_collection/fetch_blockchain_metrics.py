import requests
import csv
import time
import os
from datetime import datetime

# ──────────────────────────── Configuration ────────────────────────────
CHARTS = [
    "n-transactions-excluding-popular",
    "difficulty",
    "trade-volume",
    "miners-revenue",
    "market-cap",
    "avg-block-size",
    "n-transactions-per-block",
    "median-confirmation-time",
    "hash-rate",
    "cost-per-transaction-percent",
    "n-transactions",
]

BASE_URL = "https://api.blockchain.info/charts/{}"
START_DATE = "2019-03-02"
END_DATE = "2026-03-02"
OUTPUT_FILE = "blockchain_combined.csv"
DELAY_BETWEEN_REQUESTS = 3  # seconds, to respect rate limits

# ──────────────────────────── Helpers ──────────────────────────────────

def calculate_timespan_days(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    return (end - start).days


def fetch_chart_data(chart_name, timespan_days):
    url = BASE_URL.format(chart_name)
    params = {
        "timespan": f"{timespan_days}days",
        "start": START_DATE,
        "format": "json",
        "sampled": "false",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────── Main ─────────────────────────────────────

def main():
    timespan_days = calculate_timespan_days(START_DATE, END_DATE)

    print("=" * 65)
    print("  Blockchain.com  ·  Charts API Data Collector")
    print("=" * 65)
    print(f"  Date range : {START_DATE}  →  {END_DATE}")
    print(f"  Timespan   : {timespan_days} days")
    print(f"  Metrics    : {len(CHARTS)}")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 65, "\n")

    all_data = {}
    failed = []

    for i, chart_name in enumerate(CHARTS, 1):
        label = f"[{i:>2}/{len(CHARTS)}]"
        print(f"{label}  {chart_name} ... ", end="", flush=True)

        try:
            data = fetch_chart_data(chart_name, timespan_days)
            n = len(data["values"])
            all_data[chart_name] = data
            print(f"✓  {n:,} data points")
        except requests.exceptions.HTTPError as e:
            print(f"✗  HTTP {e.response.status_code}: {e}")
            failed.append(chart_name)
        except Exception as e:
            print(f"✗  {e}")
            failed.append(chart_name)

        if i < len(CHARTS):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # ── Build combined CSV ──
    if all_data:
        print("\nBuilding combined CSV ... ", end="", flush=True)

        combined = {}
        for chart_name, data in all_data.items():
            for entry in data["values"]:
                ts = entry["x"]
                combined.setdefault(ts, {})[chart_name] = entry["y"]

        sorted_timestamps = sorted(combined.keys())
        chart_names = list(all_data.keys())

        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["unix_timestamp", "date"] + chart_names)
            for ts in sorted_timestamps:
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                row = [ts, date_str] + [combined[ts].get(c, "") for c in chart_names]
                writer.writerow(row)

        print(f"✓  {len(sorted_timestamps):,} rows  →  {OUTPUT_FILE}")

    # ── Summary ──
    print("\n" + "=" * 65)
    print(f"  Succeeded : {len(all_data)} / {len(CHARTS)}")
    if failed:
        print(f"  Failed    : {', '.join(failed)}")
    print(f"  Output    : {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()