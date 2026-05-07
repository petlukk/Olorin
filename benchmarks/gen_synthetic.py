#!/usr/bin/env python3
"""Generate a synthetic transactions CSV for benchmarking eacrunch vs pandas.

Schema mirrors the financial-statement shape used in the README sample:
date,category,amount,merchant — covers all eacrunch column types
(text dates, text categories, numeric amounts, text merchants).

Usage: gen_synthetic.py <n_rows> <out_path>
"""
import random
import sys
from datetime import date, timedelta


def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: gen_synthetic.py <n_rows> <out_path>\n")
        sys.exit(2)
    n = int(sys.argv[1])
    out_path = sys.argv[2]

    # Deterministic so benchmarks are reproducible.
    random.seed(42)

    categories = ["groceries", "rent", "transport", "food", "utilities",
                  "entertainment", "subscriptions", "health"]
    merchants = ["Coop", "ICA", "Willys", "SL", "Landlord", "Spotify",
                 "Netflix", "Apoteket", "Restaurant", "Kiosk", "Vattenfall",
                 "Pharmacy", "Gym"]
    base = date(2024, 1, 1)

    with open(out_path, "w") as f:
        f.write("date,category,amount,merchant\n")
        for _ in range(n):
            d = base + timedelta(days=random.randint(0, 365))
            cat = random.choice(categories)
            # Realistic amount range: rent is bigger, groceries smaller.
            if cat == "rent":
                amt = round(random.uniform(900, 2000), 2)
            elif cat in ("subscriptions", "transport"):
                amt = round(random.uniform(5, 50), 2)
            else:
                amt = round(random.uniform(8, 200), 2)
            mer = random.choice(merchants)
            f.write(f"{d.isoformat()},{cat},{amt:.2f},{mer}\n")

    sys.stderr.write(f"wrote {n} rows to {out_path}\n")


if __name__ == "__main__":
    main()
