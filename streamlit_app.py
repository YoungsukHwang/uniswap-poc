# streamlit_app.py
import os, json, pandas as pd, streamlit as st
from datetime import datetime, timezone

# ---------- Optional imports (app still runs without Dune/OpenAI) ----------
try:
    from dune_client.client import DuneClient
    from dune_client.query import QueryBase
    from dune_client.types import QueryParameter
    HAVE_DUNE = True
except Exception:
    HAVE_DUNE = False

st.set_page_config(page_title="Uniswap 24h Summarizer", layout="wide")
st.title("Uniswap On-Chain Summarizer (POC) — No Filters")

# ---------- Config ----------
DEFAULT_QUERY_ID = int(os.getenv("DUNE_QUERY_ID", "0"))     # set in Secrets or env
DEFAULT_WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "24")) # used only if your Dune SQL defines {{window_hours}}

# ---------- Data loaders ----------
@st.cache_data(ttl=15 * 60)
def load_from_dune(query_id: int, window_hours: int) -> pd.DataFrame:
    """Try calling with window_hours param, retry without if Dune rejects it."""
    if not HAVE_DUNE:
        raise RuntimeError("Dune SDK not available")
    dune = DuneClient(api_key=st.secrets["DUNE_API_KEY"])
    try:
        q_with_param = QueryBase(
            query_id=query_id,
            params=[QueryParameter.number_type(name="window_hours", value=window_hours)],
        )
        return dune.run_query_dataframe(q_with_param)
    except Exception as e:
        msg = str(e).lower()
        if "unknown parameter" in msg or "unknown parameters" in msg:
            q_no_param = QueryBase(query_id=query_id)
            return dune.run_query_dataframe(q_no_param)
        raise

@st.cache_data
def load_json(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict) and "rows" in data:
        return pd.DataFrame(data["rows"])
    return pd.DataFrame(data.get("rows", []))

# ---------- Sidebar: source ----------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Choose source", ["Dune API", "Local JSON"], horizontal=True)
json_path = st.sidebar.text_input("Local JSON path", value="metrics_24h.json")
window_hours = st.sidebar.number_input("Window (hours for Dune param)", min_value=1, max_value=168, value=DEFAULT_WINDOW_HOURS)

# ---------- Load data ----------
as_of = datetime.now(timezone.utc).isoformat()
if mode == "Dune API":
    if not DEFAULT_QUERY_ID:
        st.error("Set DUNE_QUERY_ID in Streamlit Secrets or env.")
        st.stop()
    if "DUNE_API_KEY" not in st.secrets:
        st.error("Add DUNE_API_KEY in Streamlit Secrets.")
        st.stop()
    try:
        df = load_from_dune(DEFAULT_QUERY_ID, int(window_hours))
    except Exception as e:
        st.error(f"Dune API error: {e}")
        st.stop()
else:
    try:
        df = load_json(json_path)
    except Exception as e:
        st.error(f"JSON load error: {e}")
        st.stop()

if df.empty:
    st.warning("No data returned. Check your query/JSON.")
    st.stop()

# Use full dataset (no filters)
view = df.copy()

# ---------- Column detection ----------
chain_col   = "chain" if "chain" in view.columns else None
version_col = "version" if "version" in view.columns else None
vol_col     = "volume_usd" if "volume_usd" in view.columns else None

# Heuristics for swaps & traders:
# - Prefer explicit aggregated columns if present (`swaps`, `unique_traders`)
# - Else fall back to transaction-level dedup if available (`tx_hash`, `tx_from`)
# - Else mark as None (do NOT show KPI if None)
swaps_col   = "swaps" if "swaps" in view.columns else None
traders_col = "unique_traders" if "unique_traders" in view.columns else None
has_tx_hash = "tx_hash" in view.columns
has_tx_from = "tx_from" in view.columns

# ---------- KPI computation (only when available) ----------
kpi_volume = float(view[vol_col].sum()) if vol_col else None

if swaps_col:
    kpi_swaps = int(pd.to_numeric(view[swaps_col], errors="coerce").fillna(0).sum())
elif has_tx_hash:
    kpi_swaps = int(view["tx_hash"].nunique())  # per-transaction dedup
else:
    kpi_swaps = None  # unknown → don't show

if traders_col:
    kpi_traders = int(pd.to_numeric(view[traders_col], errors="coerce").fillna(0).sum())
elif has_tx_from:
    kpi_traders = int(view["tx_from"].nunique())  # unique addresses
else:
    kpi_traders = None  # unknown → don't show

# ---------- Daily Digest (LLM) ----------
st.subheader("Daily Digest (LLM)")

def fmt_usd(x):
    try:
        x = float(x)
        if x >= 1e9:  return f"${x/1e9:.2f}B"
        if x >= 1e6:  return f"${x/1e6:.2f}M"
        if x >= 1e3:  return f"${x/1e3:.2f}K"
        return f"${x:,.0f}"
    except Exception:
        return "—"

def build_metrics_payload():
    payload = {
        "as_of_utc": as_of,
        "window_hours": int(window_hours),
        "totals": {}
    }
    if kpi_volume is not None:
        payload["totals"]["volume_usd"] = round(kpi_volume, 2)
    if kpi_swaps is not None:
        payload["totals"]["swaps"] = int(kpi_swaps)
    if kpi_traders is not None:
        payload["totals"]["unique_traders"] = int(kpi_traders)

    if chain_col and version_col and vol_col:
        bc = (
            view.groupby([chain_col, version_col])[vol_col]
            .sum().reset_index()
            .rename(columns={chain_col: "chain", version_col: "version", vol_col: "volume_usd"})
            .to_dict(orient="records")
        )
        payload["by_chain_version"] = bc

    pool_label = "pool_or_pair" if "pool_or_pair" in view.columns else (
        "token_symbol" if "token_symbol" in view.columns else ("pool" if "pool" in view.columns else None)
    )
    if pool_label and vol_col:
        top = (
            view.groupby(pool_label)[vol_col]
            .sum().reset_index()
            .sort_values(vol_col, ascending=False)
            .head(5)
            .rename(columns={pool_label: "pool_or_pair", vol_col: "volume_usd"})
            .to_dict(orient="records")
        )
        payload["top_pools"] = top

    return payload

def render_digest_md(d):
    headline = d.get("headline", f"Uniswap activity update (last {int(window_hours)}h)")
    bullets  = d.get("bullets", [])
    watch    = d.get("watch_next", "")
    md = f"### {headline}\n\n"
    for b in bullets:
        md += f"- {b}\n"
    if watch:
        md += f"\n**What to watch next:** {watch}\n"
    return md

def fallback_digest_md():
    parts = [f"### Uniswap activity update (last {int(window_hours)}h)"]
    totals = []
    if kpi_volume is not None:  totals.append(f"Volume: {fmt_usd(kpi_volume)}")
    if kpi_swaps is not None:   totals.append(f"Swaps: {kpi_swaps:,}")
    if kpi_traders is not None: totals.append(f"Unique traders: {kpi_traders:,}")
    if totals: parts.append("- " + " • ".join(totals))
    if chain_col and version_col and vol_col:
        parts.append("- See breakdown by chain/version below.")
    parts.append("- Top pools shown in the chart below.")
    parts.append("**What to watch next:** monitor unusual deltas vs prior window.")
    return "\n".join(parts)

use_llm = st.sidebar.checkbox("Use OpenAI (if key set)", value=True)

rendered_md = fallback_digest_md()
if use_llm and "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        metrics = build_metrics_payload()

        # Prefer Chat Completions with JSON mode (broad SDK compatibility)
        system = (
            "You are a product-minded analyst. Return ONLY JSON with keys: "
            "headline (string), bullets (array of 2-4 strings), watch_next (string, optional). "
            "Use ONLY the provided JSON metrics; do not invent numbers."
        )
        user = f"""Metrics JSON:
{json.dumps(metrics)}"""

        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content": system},
                {"role":"user","content": user}
            ],
            response_format={"type":"json_object"}
        )

        raw = comp.choices[0].message.content
        dig = json.loads(raw) if raw else {}
        if isinstance(dig, dict):
            rendered_md = render_digest_md(dig)
        else:
            rendered_md = fallback_digest_md()

    except Exception as e:
        st.warning(f"LLM disabled (error: {e}). Showing fallback.")
        rendered_md = fallback_digest_md()

# Render the digest nicely (no code block)
st.markdown(rendered_md)

# ---------- KPI strip (only show what’s real) ----------
st.markdown("---")
cols = []
if kpi_volume is not None:  cols.append(("Volume (USD, window)", fmt_usd(kpi_volume)))
if kpi_swaps  is not None:  cols.append(("Swaps (window)", f"{kpi_swaps:,}"))
if kpi_traders is not None: cols.append(("Unique traders (window)", f"{kpi_traders:,}"))

if cols:
    cols_stream = st.columns(len(cols))
    for c, (label, value) in zip(cols_stream, cols):
        c.metric(label, value)
st.caption(f"As of (UTC): {as_of} • Window context: {int(window_hours)}h (if paramized in SQL)")

# ---------- Charts ----------
st.subheader("Volume by Chain & Version")
if chain_col and version_col and vol_col:
    grp = view.groupby([chain_col, version_col])[vol_col].sum().reset_index()
    pivot = grp.pivot(index=chain_col, columns=version_col, values=vol_col).fillna(0)
    st.bar_chart(pivot)
else:
    st.info("Need columns: chain, version, volume_usd")

st.subheader("Top 10 Pools by Volume")
pool_label = (
    "pool_or_pair"
    if "pool_or_pair" in view.columns
    else ("token_symbol" if "token_symbol" in view.columns else ("pool" if "pool" in view.columns else None))
)
if pool_label and vol_col:
    top = view.groupby(pool_label)[vol_col].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top)
else:
    st.info("Need columns: pool (or token_symbol) + volume_usd")

# ---------- Raw Table ----------
st.divider()
st.subheader("Raw Table (full dataset)")
st.dataframe(view, use_container_width=True)
