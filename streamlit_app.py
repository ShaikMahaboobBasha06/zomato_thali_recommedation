"""
streamlit_app.py
-----------------
Streamlit front end for the Thali Trail Apriori food-pairing recommender.

Set this file as the "Main file path" in Streamlit Cloud (not generate_rules.py
— that's an offline helper script, not a web app).

Unlike the first version of this app (which read precomputed rules.json),
this one runs the Apriori pipeline live, in-process, using mlxtend — so you
can tune min_support / min_confidence from the sidebar and watch the rules
change, and add synthetic orders that get folded into the training data on
the fly. It falls back to the precomputed data/rules.json data if mlxtend
isn't available for some reason.
"""

import json
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Thali Trail — Apriori Food Pairing Recommender",
    page_icon="🍽️",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

CUSTOM_CSS = """
<style>
:root{
  --chili:#D3452B; --chili-soft:#7A2C1E;
  --turmeric:#E8A33D; --turmeric-soft:#6E4C1D;
  --basil:#4C9A6A; --basil-soft:#24402F;
  --line:#3A322A; --muted:#B0A08D; --faint:#7C6F5F;
}
.block-container{ padding-top: 2rem; max-width: 1240px; }
.dish-card{
  background:#221C17; border:1px solid var(--line); border-radius:18px;
  padding:14px; height:100%;
}
.plate-wrap{
  width:100%; aspect-ratio:1/1; border-radius:50%; overflow:hidden;
  background:#2C241D; border:3px solid #2C241D; box-shadow:0 0 0 1px var(--line);
  margin-bottom:10px; display:flex; align-items:center; justify-content:center;
}
.plate-wrap img{ width:100%; height:100%; object-fit:cover; display:block; }
.plate-wrap .emoji{ font-size:2.4rem; }
.dish-name{ font-weight:700; font-size:1.02rem; margin-bottom:2px; }
.dish-cat{ font-family:monospace; font-size:0.68rem; text-transform:uppercase;
  letter-spacing:.05em; color:var(--faint); margin-bottom:6px; }
.dish-desc{ font-size:0.82rem; color:var(--muted); min-height:2.6em; }
.dish-foot{ display:flex; justify-content:space-between; margin-top:8px;
  font-family:monospace; font-size:0.85rem; }
.price{ color:var(--turmeric); font-weight:700; }
.rating{ color:var(--muted); }

.reco-card{ background:#221C17; border:1px solid var(--line); border-radius:14px;
  padding:14px 16px; margin-bottom:10px; }
.reco-top{ display:flex; justify-content:space-between; align-items:baseline; }
.reco-name{ font-weight:700; }
.reco-because{ font-family:monospace; font-size:0.68rem; color:var(--faint); margin:2px 0 8px; }
.reco-because b{ color:var(--basil); }
.lift-badge{ font-family:monospace; font-size:0.62rem; font-weight:700; padding:2px 7px;
  border-radius:999px; background:var(--basil-soft); color:var(--basil); margin-left:6px; }
.trail{ height:6px; border-radius:999px; background:#2C241D; overflow:hidden; margin-bottom:6px; }
.trail-fill{ height:100%; background:linear-gradient(90deg,var(--chili),var(--turmeric)); }
.reco-stats{ font-family:monospace; font-size:0.68rem; color:var(--faint); display:flex; gap:10px; }
.reco-stats b{ color:#F6EEE1; }

.eyebrow{ font-family:monospace; text-transform:uppercase; letter-spacing:.08em;
  font-size:0.72rem; color:var(--turmeric); margin-bottom:6px; }

.cart-total{ font-family:monospace; font-size:1.4rem; font-weight:700; color:var(--turmeric); }
.badge-new{ font-family:monospace; font-size:0.6rem; font-weight:700; padding:2px 7px;
  border-radius:999px; background:var(--turmeric-soft); color:var(--turmeric); margin-left:6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(name: str):
    with open(DATA_DIR / name, "r") as f:
        return json.load(f)


products = load_json("products.json")
base_transactions = load_json("transactions.json")
products_by_name = {p["name"]: p for p in products}
ALL_ITEM_NAMES = sorted(products_by_name.keys())

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "selected" not in st.session_state:
    st.session_state.selected = set()
if "custom_orders" not in st.session_state:
    st.session_state.custom_orders = []  # list[list[str]] added live in the "Grow the dataset" tab


def toggle(name: str):
    if name in st.session_state.selected:
        st.session_state.selected.discard(name)
    else:
        st.session_state.selected.add(name)


def image_html(product: dict, size: int = 140) -> str:
    """<img> with a two-step fallback: primary image -> fallback image -> emoji."""
    img_id = f"img-{product['id']}"
    return f"""
    <div class="plate-wrap">
      <img id="{img_id}" src="{product['image']}" width="{size}" height="{size}"
           onerror="
             if(this.dataset.stage!=='fallback'){{
               this.dataset.stage='fallback'; this.src='{product['fallbackImage']}';
             }} else {{
               this.replaceWith(Object.assign(document.createElement('span'),
                 {{className:'emoji', textContent:'{product['emoji']}'}}));
             }}
           " />
    </div>
    """


# ---------------------------------------------------------------------------
# Live Apriori engine — runs mlxtend in-process so the sidebar sliders and
# the "grow the dataset" tab actually change the mined rules in real time.
# Falls back to the precomputed data/rules.json if mlxtend is unavailable.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def mine_rules(transactions: tuple, min_support: float, min_confidence: float):
    try:
        import pandas as pd
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        rules = load_json("rules.json")
        itemsets = load_json("frequent_itemsets.json")
        return rules, itemsets

    tx = [list(t) for t in transactions]
    te = TransactionEncoder()
    encoded = te.fit(tx).transform(tx)
    df = pd.DataFrame(encoded, columns=te.columns_)

    frequent = apriori(df, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return [], []

    itemsets_out = [
        {"items": sorted(list(row["itemsets"])), "support": round(float(row["support"]), 4)}
        for _, row in frequent.sort_values("support", ascending=False).iterrows()
    ]

    try:
        rules_df = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    except ValueError:
        return [], itemsets_out

    rules_out = [
        {
            "antecedents": sorted(list(row["antecedents"])),
            "consequents": sorted(list(row["consequents"])),
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
        }
        for _, row in rules_df.sort_values("confidence", ascending=False).iterrows()
    ]
    return rules_out, itemsets_out


def compute_recommendations(selected: set, rules: list) -> list:
    if not selected:
        return []
    best = {}
    for rule in rules:
        if not all(a in selected for a in rule["antecedents"]):
            continue
        for cons in rule["consequents"]:
            if cons in selected:
                continue
            current = best.get(cons)
            if current is None or rule["confidence"] > current["confidence"]:
                best[cons] = {**rule, "name": cons, "because": rule["antecedents"]}
    return sorted(best.values(), key=lambda r: (-r["confidence"], -r["lift"]))


# ---------------------------------------------------------------------------
# Sidebar — model controls + filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model controls")
    min_support = st.slider("Minimum support", 0.05, 0.60, 0.20, 0.05)
    min_confidence = st.slider("Minimum confidence", 0.10, 1.00, 0.50, 0.05)
    st.caption("Raise support to keep only very common combos; raise confidence to keep only strong pairings.")

    st.markdown("---")
    st.markdown("### 🔍 Menu filters")
    search_query = st.text_input("Search dishes", placeholder="e.g. biryani, pizza…")
    category_filter = st.multiselect(
        "Categories", sorted({p["category"] for p in products}), default=[]
    )
    price_max = max(p["price"] for p in products)
    price_range = st.slider("Max price (₹)", 0, price_max, price_max, 10)

    st.markdown("---")
    n_custom = len(st.session_state.custom_orders)
    st.caption(
        f"Training on **{len(base_transactions) + n_custom} orders** "
        f"({len(base_transactions)} original"
        + (f" + {n_custom} added this session)" if n_custom else ")")
    )
    if n_custom and st.button("Reset session-added orders"):
        st.session_state.custom_orders = []
        st.rerun()

all_transactions = tuple(tuple(t) for t in base_transactions + st.session_state.custom_orders)
rules, itemsets = mine_rules(all_transactions, min_support, min_confidence)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="eyebrow">LIVE APRIORI ENGINE · ADJUST THRESHOLDS IN THE SIDEBAR</div>',
    unsafe_allow_html=True,
)
st.title("Order one dish. See what usually rides along. 🍽️")
st.caption(
    "Tap the dishes you're ordering — the engine mines historical baskets for pairings "
    "that show up together often enough to matter, ranked by confidence and lift."
)

tab_order, tab_insights, tab_network, tab_grow = st.tabs(
    ["🍽️ Order & Recommend", "📊 Insights", "🕸️ Association Network", "➕ Grow the Dataset"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Order & Recommend
# ---------------------------------------------------------------------------
with tab_order:
    visible_products = products
    if search_query:
        q = search_query.lower()
        visible_products = [
            p for p in visible_products if q in p["name"].lower() or q in p["category"].lower()
        ]
    if category_filter:
        visible_products = [p for p in visible_products if p["category"] in category_filter]
    visible_products = [p for p in visible_products if p["price"] <= price_range]

    top_row_left, top_row_right = st.columns([5, 2])
    with top_row_left:
        if st.session_state.selected:
            st.info("Selected: " + ", ".join(sorted(st.session_state.selected)))
        else:
            st.caption("No dishes selected yet.")
    with top_row_right:
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🎲 Surprise me", use_container_width=True, help="Pick a real popular combo from the data"):
                combo_pool = [i for i in itemsets if len(i["items"]) >= 2] or itemsets
                if combo_pool:
                    import random

                    choice = random.choice(combo_pool[: max(3, len(combo_pool) // 2)])
                    st.session_state.selected = set(choice["items"])
                    st.rerun()
        with bcol2:
            if st.session_state.selected and st.button("Clear all", use_container_width=True):
                st.session_state.selected.clear()
                st.rerun()

    if st.session_state.selected:
        cart_total = sum(products_by_name[n]["price"] for n in st.session_state.selected if n in products_by_name)
        st.markdown(f'<span class="cart-total">Cart total: ₹{cart_total}</span>', unsafe_allow_html=True)

    st.divider()
    st.subheader(f"Menu · {len(visible_products)} dish{'es' if len(visible_products) != 1 else ''}")

    if not visible_products:
        st.warning("No dishes match your filters — try widening the search, categories, or price cap.")

    cols_per_row = 4
    rows = [visible_products[i : i + cols_per_row] for i in range(0, len(visible_products), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, product in zip(cols, row):
            with col:
                is_selected = product["name"] in st.session_state.selected
                st.markdown(f'<div class="dish-card">{image_html(product)}', unsafe_allow_html=True)
                st.markdown(
                    f"""<div class="dish-name">{product['name']}</div>
                        <div class="dish-cat">{product['category']}</div>
                        <div class="dish-desc">{product['description']}</div>
                        <div class="dish-foot">
                          <span class="price">₹{product['price']}</span>
                          <span class="rating">★ {product['rating']:.1f}</span>
                        </div>
                        </div>""",
                    unsafe_allow_html=True,
                )
                st.button(
                    "✓ Selected" if is_selected else "Select",
                    key=f"btn-{product['id']}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                    on_click=toggle,
                    args=(product["name"],),
                )

    st.divider()
    st.subheader("Pairing trail")

    recos = compute_recommendations(st.session_state.selected, rules)
    if not st.session_state.selected:
        st.caption("Select a dish above — this fills in with what the algorithm recommends alongside it.")
    elif not recos:
        st.warning(
            f"No rule clears {int(min_confidence * 100)}% confidence at {min_support:.2f} support for this "
            "combination — try lowering the thresholds in the sidebar."
        )
    else:
        st.caption(f"Based on {len(st.session_state.selected)} selected dish(es), ranked by confidence.")
        reco_cols = st.columns(2)
        for i, r in enumerate(recos):
            product = products_by_name.get(r["name"])
            if not product:
                continue
            conf_pct = round(r["confidence"] * 100)
            trending = '<span class="lift-badge">trending pair</span>' if r["lift"] > 1.2 else ""
            with reco_cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="reco-card">
                      <div class="reco-top">
                        <span class="reco-name">{product['name']}</span>
                        <span class="price">₹{product['price']}</span>
                      </div>
                      <div class="reco-because">because you picked <b>{' + '.join(r['because'])}</b>{trending}</div>
                      <div class="trail"><div class="trail-fill" style="width:{conf_pct}%"></div></div>
                      <div class="reco-stats">
                        <span>conf <b>{conf_pct}%</b></span>
                        <span>support <b>{r['support']:.2f}</b></span>
                        <span>lift <b>{r['lift']:.2f}</b></span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    f"+ Add {product['name']}",
                    key=f"add-{product['id']}",
                    on_click=toggle,
                    args=(product["name"],),
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# TAB 2 — Insights dashboard
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Model at a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training orders", len(all_transactions))
    m2.metric("Frequent itemsets", len(itemsets))
    m3.metric("Association rules", len(rules))
    avg_conf = sum(r["confidence"] for r in rules) / len(rules) if rules else 0
    m4.metric("Avg. rule confidence", f"{avg_conf * 100:.0f}%")

    st.divider()
    st.subheader("Most frequent combinations")
    st.caption("Support of the top item combinations in the current training data.")
    if itemsets:
        top_itemsets = sorted(itemsets, key=lambda t: -t["support"])[:12]
        chart_data = {" + ".join(t["items"]): t["support"] for t in top_itemsets}
        st.bar_chart(chart_data, horizontal=True)
    else:
        st.info("No itemsets clear the current minimum support — lower it in the sidebar.")

    st.divider()
    st.subheader("Full association rules")
    if rules:
        table_rows = [
            {
                "If ordered": " + ".join(r["antecedents"]),
                "Then recommend": " + ".join(r["consequents"]),
                "Support": r["support"],
                "Confidence": r["confidence"],
                "Lift": r["lift"],
            }
            for r in rules
        ]
        st.dataframe(
            table_rows,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.0f%%"
                ),
                "Support": st.column_config.NumberColumn("Support", format="%.2f"),
                "Lift": st.column_config.NumberColumn("Lift", format="%.2f"),
            },
            hide_index=True,
        )
        st.download_button(
            "⬇ Download rules as JSON",
            data=json.dumps(rules, indent=2),
            file_name="thali_trail_rules.json",
            mime="application/json",
        )
    else:
        st.info("No rules clear the current thresholds — lower min support / min confidence in the sidebar.")

# ---------------------------------------------------------------------------
# TAB 3 — Association network graph
# ---------------------------------------------------------------------------
with tab_network:
    st.subheader("How dishes connect")
    st.caption("Each edge is a rule; thicker/brighter edges have higher confidence. Node size reflects support.")

    if not rules:
        st.info("No rules to draw at the current thresholds — lower them in the sidebar.")
    else:
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.DiGraph()
            support_by_item = {
                i["items"][0]: i["support"] for i in itemsets if len(i["items"]) == 1
            }
            for name in support_by_item:
                G.add_node(name)
            for r in rules:
                for a in r["antecedents"]:
                    for c in r["consequents"]:
                        if G.has_edge(a, c):
                            if r["confidence"] > G[a][c]["weight"]:
                                G[a][c]["weight"] = r["confidence"]
                        else:
                            G.add_edge(a, c, weight=r["confidence"])

            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor("#171310")
            ax.set_facecolor("#171310")

            pos = nx.spring_layout(G, seed=7, k=1.1)
            node_sizes = [2200 + support_by_item.get(n, 0.1) * 6000 for n in G.nodes()]
            edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=node_sizes, node_color="#2C241D", edgecolors="#E8A33D", linewidths=2
            )
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#F6EEE1")
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                width=[1 + w * 5 for w in edge_weights],
                edge_color="#D3452B",
                alpha=0.75,
                arrowsize=16,
                connectionstyle="arc3,rad=0.08",
            )
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
        except ImportError:
            st.warning(
                "Install `networkx` and `matplotlib` (already in requirements.txt) to see the network graph."
            )

# ---------------------------------------------------------------------------
# TAB 4 — Grow the dataset
# ---------------------------------------------------------------------------
with tab_grow:
    st.subheader("Add a synthetic order")
    st.caption(
        "Simulate a new customer order — it's folded into the training data immediately, "
        "so you can watch new rules appear (or existing ones strengthen/weaken) live. "
        "Session-only: nothing is written back to the repo."
    )

    with st.form("add_order_form", clear_on_submit=True):
        new_order = st.multiselect("Items in this order", ALL_ITEM_NAMES)
        submitted = st.form_submit_button("Add order to training data")
        if submitted:
            if len(new_order) < 2:
                st.error("An order needs at least 2 items to contribute to pairing rules.")
            else:
                st.session_state.custom_orders.append(new_order)
                st.success(f"Added: {' + '.join(new_order)}")
                st.rerun()

    if st.session_state.custom_orders:
        st.divider()
        st.markdown(f"**{len(st.session_state.custom_orders)} order(s) added this session:**")
        for i, order in enumerate(st.session_state.custom_orders):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"{i + 1}. " + " + ".join(order) + ' <span class="badge-new">NEW</span>', unsafe_allow_html=True)
            if c2.button("Remove", key=f"remove-order-{i}"):
                st.session_state.custom_orders.pop(i)
                st.rerun()
    else:
        st.caption("No session orders added yet.")

st.divider()
st.caption("Built on a live Apriori pairing model · demo data only, not affiliated with Zomato.")
