"""
Build the static dashboard.html by injecting the latest dashboard_data.json
into dashboard_template.html. Run export_dashboard.py first if the data
needs refreshing.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "dashboard_template.html"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dashboard_data.json"
OUTPUT_PATH = PROJECT_ROOT / "dashboard.html"

if __name__ == "__main__":
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = DATA_PATH.read_text(encoding="utf-8")

    output = template.replace("__DASHBOARD_DATA__", data_json)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(output):,} bytes)")
