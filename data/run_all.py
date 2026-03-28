"""
Run all data-generation scripts in the correct order.
Usage:
    python data/run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SCRIPTS = [
    ("Merchant profiles → MongoDB",       ROOT / "data" / "generate_merchants.py"),
    ("Transaction history → PostgreSQL",  ROOT / "data" / "generate_transactions.py"),
]

for label, script in SCRIPTS:
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    result = subprocess.run([sys.executable, str(script)])
    if result.returncode != 0:
        print(f"\n[ERROR] {script.name} failed. Stopping.")
        sys.exit(1)

print(f"\n{'═' * 55}")
print("  All data generated successfully!")
print(f"{'═' * 55}")
print("\nNext step — start the mock bureau API in a separate terminal:")
print("  python data/mock_bureau_api.py")
