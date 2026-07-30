# 🍽️ Thali Trail — Apriori Food Pairing Recommender

An interactive food-recommendation app built on top of the market-basket
analysis from `Zomato_Food_Recommendation_Apriori.ipynb`. Pick the dishes
you're ordering and the app tells you what pairs with them, using
association rules mined with the **Apriori algorithm**.

> This started as a data science notebook (`mlxtend` + Apriori on 20 sample
> orders). This repo turns that analysis into a real, deployable product:
> a menu you can click through, with recommendations ranked by confidence
> and lift, plus the full rules table for anyone who wants to see the math.

The app ships in **two forms** that share the same data and logic:

| | Entry point | Runs on |
|---|---|---|
| **Streamlit app** (recommended) | `streamlit_app.py` | Streamlit Community Cloud |
| Static site | `index.html` | GitHub Pages / Netlify / Vercel |

### 🆕 Advanced mode (Streamlit app)

The Streamlit app now runs **Apriori live**, in-process, instead of just
reading a precomputed rules file:

- **Sidebar model controls** — sliders for minimum support and minimum
  confidence recompute the rules on the spot, so you can watch pairings
  appear/disappear as you loosen or tighten the thresholds.
- **Menu filters** — search box, category multiselect, and a max-price
  slider on the menu.
- **Cart total + "Surprise me"** — running total for your selection, and a
  button that jumps to a real popular combo pulled from the data.
- **Insights tab** — key metrics (orders / itemsets / rules / avg.
  confidence), a frequent-itemsets chart, the full rules table, and a JSON
  download button.
- **Association Network tab** — a graph of dishes as nodes and rules as
  directed edges, edge thickness = confidence, node size = support.
- **Grow the Dataset tab** — add a synthetic order from the UI and it's
  folded into the training data immediately (session-only, nothing is
  written back to the repo) — the fastest way to see how Apriori reacts to
  new data.

---

## ✨ What it does

- **Select dishes** from a 10-item menu (biryanis, sides, pizza, fast food,
  beverages) — each with a photo, price, rating and description.
- **Get live recommendations**: as soon as you select something, the app
  matches your selection against the mined association rules and shows what
  usually gets ordered alongside it, ranked by confidence.
- **See why**: every recommendation shows *"because you picked X"*, plus its
  support, confidence, and lift — and a "trending pair" badge for
  high-lift combinations.
- **Inspect the model**: an expander/toggle reveals the full association-
  rules table and a bar chart of the most frequent item combinations in the
  training data — the same numbers the original notebook produced.
- **One-tap add**: add a recommended dish straight to your selection and see
  what pairs with *that*.

---

## 🚀 Deploy on Streamlit Community Cloud

This is the entry point to use for Streamlit — **not** `generate_rules.py`
(that's just an offline helper script, not a web app; pointing Streamlit at
it is what causes a `SyntaxError`).

1. Push this repo to GitHub (all files, including `streamlit_app.py`,
   `.streamlit/config.toml`, and the `data/` folder).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick your repo and branch.
4. Set **Main file path** to:
   ```
   streamlit_app.py
   ```
5. Deploy. Streamlit Cloud installs `requirements.txt` automatically
   (`streamlit`, `pandas`, `mlxtend`, `networkx`, `matplotlib` — the app
   mines the Apriori rules live rather than only reading precomputed JSON).

If you already created an app pointing at the wrong file: open your app on
Streamlit Cloud → **Manage app** (bottom right) → **Settings** → change
**Main file path** to `streamlit_app.py` → **Save**, then reboot the app.

**Run it locally first (recommended before deploying):**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🧠 How the recommendations are computed

1. `notebooks/Zomato_Food_Recommendation_Apriori.ipynb` — the original
   analysis: 20 sample orders → `TransactionEncoder` → `apriori()`
   (min support 0.20) → `association_rules()` (min confidence 0.50).
2. `generate_rules.py` reproduces that exact pipeline in a plain script and
   writes the results to `data/rules.json` and `data/frequent_itemsets.json`.
   This is an **offline, one-time step** — run it yourself when the data
   changes, it is not run by the deployed app.
3. `streamlit_app.py` (and, for the static version, `app.js`) loads those
   JSON files and, for the currently selected dishes, finds every rule
   whose antecedents are fully contained in the selection, collects the
   consequents, and ranks them by confidence (ties broken by lift) — the
   same logic as the notebook's `recommend(item)`, generalized to multiple
   selected items.

---

## 🗂️ Project structure

```
.
├── streamlit_app.py             # Streamlit entry point (deploy this)
├── .streamlit/config.toml       # Streamlit theme
├── requirements.txt             # streamlit — installed by Streamlit Cloud
│
├── index.html                   # Static-site entry point (alternative)
├── style.css / app.js           # Static-site design + logic
│
├── data/
│   ├── transactions.json        # Source orders (from the notebook)
│   ├── products.json            # Menu metadata: name, price, image, etc.
│   ├── rules.json                # Generated association rules
│   └── frequent_itemsets.json    # Generated frequent itemsets
│
├── generate_rules.py            # Reproduces the notebook's Apriori pipeline
├── requirements-dataprep.txt    # pandas + mlxtend, for generate_rules.py only
├── notebooks/
│   └── Zomato_Food_Recommendation_Apriori.ipynb   # Original analysis
│
├── .github/workflows/deploy.yml # GitHub Pages auto-deploy (static site)
├── vercel.json / netlify.toml   # One-click static deploy configs
└── README.md
```

---

## ☁️ Alternative: deploy the static site

If you'd rather not run Streamlit, `index.html` + `app.js` + `style.css` is
a complete, framework-free app that reads the same `data/` JSON files
client-side.

**GitHub Pages (included workflow)**
1. Push this repo to GitHub.
2. In **Settings → Pages**, set the source to **GitHub Actions**.
3. Push to `main` — `.github/workflows/deploy.yml` builds and publishes
   automatically. Live at `https://<your-username>.github.io/<repo-name>/`.

**Netlify** — New site from Git → pick this repo → auto-detects
`netlify.toml` → Deploy.

**Vercel** — import the repo in the dashboard, `vercel.json` is already
set up for a static deploy.

**Run it locally:**
```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

---

## 🔧 Updating the data

**Add or edit menu items** — edit `data/products.json`. Each product needs:
`id`, `name` (must match the names used in `transactions.json`), `category`,
`price`, `rating`, `description`, `image`, `fallbackImage`, `emoji` (used if
both images fail to load).

**Add more historical orders** — edit `data/transactions.json`, then
regenerate the rules:

```bash
pip install -r requirements-dataprep.txt
python generate_rules.py
```

This overwrites `data/rules.json` and `data/frequent_itemsets.json` using
the same `min_support=0.20`, `min_confidence=0.5` thresholds as the
notebook — tune those constants at the top of `generate_rules.py` if you
want looser or stricter rules. Commit the regenerated JSON files; both the
Streamlit app and the static site read them directly and don't recompute
anything at runtime.

---

## 🎨 Design notes

The visual language ("Thali Trail") deliberately avoids food-delivery
boilerplate: a charcoal base (`#171310`) with chili red, turmeric gold, and
basil green accents; circular "plate" product images; monospace type for
every data point (price, support, confidence, lift) — so the numbers read
as *computed*, not decorative. Confidence is shown as an actual filled bar
(the "pairing trail"), sized to the real value, not just a color chip.

---

## 🛠️ Tech stack

- **Streamlit** for the primary deployable app (`streamlit_app.py`), with
  `pandas` + `mlxtend` running Apriori live and `networkx` + `matplotlib`
  rendering the association network graph
- Vanilla HTML/CSS/JS for the static-site alternative — no framework, no
  build step (reads the precomputed `data/rules.json` instead)
- `generate_rules.py` — an optional offline script for pre-baking a
  rules.json snapshot (used by the static site; the Streamlit app doesn't
  need it)
- Product photos loaded from a public image proxy at runtime, with an
  automatic emoji fallback if an image fails to load

---

## 📄 License

MIT — see [`LICENSE`](LICENSE). Demo data only; not affiliated with Zomato.
