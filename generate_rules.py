"""
generate_rules.py
------------------
Reproduces the Apriori market-basket analysis from the original
Zomato_Food_Recommendation_Apriori.ipynb notebook and exports the
results as static JSON files that the web app consumes.

Run this whenever the transaction data changes:
    python generate_rules.py
"""

import json
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

MIN_SUPPORT = 0.20
MIN_CONFIDENCE = 0.5

with open("data/transactions.json") as f:
    transactions = json.load(f)

print(f"Total transactions: {len(transactions)}")

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
frequent_itemsets.sort_values(by="support", ascending=False, inplace=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules.sort_values(by="confidence", ascending=False, inplace=True)

rules_out = []
for _, row in rules.iterrows():
    rules_out.append({
        "antecedents": sorted(list(row["antecedents"])),
        "consequents": sorted(list(row["consequents"])),
        "support": round(float(row["support"]), 4),
        "confidence": round(float(row["confidence"]), 4),
        "lift": round(float(row["lift"]), 4),
    })

itemsets_out = []
for _, row in frequent_itemsets.iterrows():
    itemsets_out.append({
        "items": sorted(list(row["itemsets"])),
        "support": round(float(row["support"]), 4),
    })

with open("data/rules.json", "w") as f:
    json.dump(rules_out, f, indent=2)

with open("data/frequent_itemsets.json", "w") as f:
    json.dump(itemsets_out, f, indent=2)

print(f"Frequent itemsets: {len(itemsets_out)}")
print(f"Association rules: {len(rules_out)}")
print("Wrote data/rules.json and data/frequent_itemsets.json")
