"""Exports schema.SCHEMA as a data dictionary (CSV + Markdown) for docs."""
import pandas as pd
from schema import SCHEMA

rows = []
for col, meta in SCHEMA.items():
    rows.append({"column": col, "dtype": meta["dtype"], "block": meta["block"],
                 "ml_role": meta["ml_role"], "description": meta["description"]})
df = pd.DataFrame(rows)
df.to_csv("data/data_dictionary.csv", index=False)

with open("data/data_dictionary.md", "w") as f:
    f.write("# PhantomGuard Synthetic Dataset — Data Dictionary\n\n")
    for block in ["key", "transaction", "user", "network", "genai", "aux", "label"]:
        sub = df[df.block == block]
        if sub.empty:
            continue
        title = {"key": "Identifiers", "transaction": "Transaction-level features",
                  "user": "User-level features", "network": "Network-level features",
                  "genai": "GenAI-specific features", "aux": "Attack-specific auxiliary features",
                  "label": "Labels & metadata (exclude from model input, except is_fraud as target)"}[block]
        f.write(f"## {title}\n\n")
        f.write("| Column | Type | Description |\n|---|---|---|\n")
        for _, r in sub.iterrows():
            f.write(f"| `{r['column']}` | {r['dtype']} | {r['description']} |\n")
        f.write("\n")

print("Wrote data/data_dictionary.csv and data/data_dictionary.md")
print(df.groupby("block").size())
