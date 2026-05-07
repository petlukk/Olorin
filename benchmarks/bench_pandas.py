#!/usr/bin/env python3
"""Pandas equivalent of `/rune eacrunch <path>` — the closest pandas
analogue of what the rune produces.

eacrunch reports per-column: type (numeric/text), and either
(count/mean/min/max/sum) for numerics or (unique count + top-3) for text.
We reproduce that here so the comparison is apples-to-apples on output
content, not just on "did the file load."

Usage: bench_pandas.py <path>
"""
import sys
import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: bench_pandas.py <path>\n")
        sys.exit(2)
    path = sys.argv[1]

    df = pd.read_csv(path)
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            print(f"{col} (number): "
                  f"count={int(s.count())}, "
                  f"mean={s.mean():.2f}, "
                  f"min={s.min():.2f}, "
                  f"max={s.max():.2f}, "
                  f"sum={s.sum():.2f}")
        else:
            counts = s.value_counts().head(3)
            top = ", ".join(str(v) for v in counts.index.tolist())
            print(f"{col} (text): {s.nunique()} unique; top values: {top}")


if __name__ == "__main__":
    main()
