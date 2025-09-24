import os, json, pandas as pd, streamlit as st
from datetime import datetime, timezone

# Optional deps guarded so app works without them locally
try:
    from dune_client.client import DuneClient
    from dune_client.query import QueryBase
    from dune_client.types import QueryParameter
    HAVE_DUNE = True
except Exception:
    HAVE_DUNE = False

st.set_page_config(page_title="Uniswap 24h Summarizer", layout="wide")
st.title("Uniswap On-Chain 24h Summarizer (POC)")

# ---------- Config ----------
DEFAULT_QUERY_ID = int(os.getenv("DUNE_QUERY_ID", "0"))  # set in secrets or env
WINDOW_HOURS     = int(os.getenv("WINDOW_HOURS", "24"))

# ---------- Data loading ----------
@st.cache_data(ttl=15 * 60)
def load_from_dune(query_id: int, params: dict | None = None) -> pd.DataFrame:
    if not HAVE_DUNE:
        st.stop()
    dune = DuneClient(api_key=st.secrets["DUNE_API_KEY"])
    qparams = []
    if params:
        for k, v in params.items():
            # text_type works well; use number_type if your SQL expects numbers
            qparams.append(QueryParameter.text_type(name=k, value=str(v)))
    query = QueryBase(query_id=query_id, params=qparams)
    return dune.run_query_dataframe(query)

@st.cache_data
def load_json(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict) and "rows" in data:
        return pd.DataFrame(data["rows"])
    return pd.DataFrame(data.get("rows", []))

st.sidebar.header("Data source")
mode = st.sidebar.radio("Choose source", ["Dune API", "Local JSON"], horizontal=True)
json_path = st.sidebar.text_input("Local JSON path", value="metrics_24h.json", help="Used when 'Local JSON' is selected")

df = pd.DataFrame()
as_of = datetime.now(timezone.utc).isoformat()
if mode == "Dune API":
    if not DEFAULT_QUERY_ID:
        st.error("Set DUNE_QUERY_ID in Streamlit Secrets or env.")
        st.stop()
    try:
        df = load_from_dune(DEFAULT_QUERY_ID, params={"window_hours": WINDOW_HOURS})
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
    st.warning("No data returned. Check your query or JSON.")
    st.stop()

# ---------- Filters ----------
chain_col   = "chain" if "chain" in df.columns else None
version_col = "version" if "version" in df.columns else None
vol_col     = "volume_usd" if "volume_usd" in df.columns else None
swaps_col   = "swaps" if "swaps" in df.columns else None
tr_col      = "unique_traders" if "unique_traders" in df.columns else None

chains = sorted(df[chain_col].dropna().unique()) if chain_col else []
versions = sorted(df[version_col].dropna().unique()) if version_col else []

sel_chains = st.sidebar.multiselect("Chain", chains, default=chains[:3] if chains else [])
sel_vers   = st.sidebar.multiselect("Version", versions, default=versions if versions else [])

mask = pd.Series([True]*len(df))
if sel_chains and chain_col: mask &= df[chain_col].isin(sel_chains)
if sel_vers and version_col: mask &= df[version_col].isin(sel_vers)
view = df[mask].copy()

# ---------- KPIs ----------
kpi_volume  = float(view[vol_col].sum()) if vol_col else 0.0
kpi_swaps   = int(view[swaps_col].sum()) if swaps_col else len(view)
kpi_traders = int(view[tr_col].sum()) if tr_col else (view["tx_from"].nunique() if "tx_from" in view else 0)

c1, c2, c3 = st.columns(3)
c1.metric("Volume (24h, USD)", f"{kpi_volume:,.0f}")
c2.metric("Swaps (24h)", f"{kpi_swaps:,}")
c3.metric("Unique traders (24h)", f"{kpi_traders:,}")
st.caption(f"As of (UTC): {as_of}")

# ---------- Charts ----------
st.subheader("Volume by Chain & Version")
if chain_col and version_col and vol_col:
    grp = view.groupby([chain_col, version_col])[vol_col].sum().reset_index()
    pivot = grp.pivot(index=chain_col, columns=version_col, values=vol_col).fillna(0)
    st.bar_chart(pivot)
else:
    st.info("Need columns: chain, version, volume_usd")

st.subheader("Top 10 Pools by 24h Volume")
pool_label = "pool_or_pair" if "pool_or_pair" in view.columns else ("token_symbol" if "token_symbol" in view.columns else ("pool" if "pool" in view.columns else None))
if pool_label and vol_col:
    top = view.groupby(pool_label)[vol_col].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top)
else:
    st.info("Need columns: pool (or token_symbol) + volume_usd")

# ---------- LLM digest (optional) ----------
st.subheader("Daily Digest (LLM)")
def fallback_digest():
    return (
        f"Headline: Uniswap 24h activity update\n"
        f"- Overall: volume ≈ ${kpi_volume:,.0f}; swaps ≈ {kpi_swaps:,}; unique traders ≈ {kpi_traders:,}\n"
        f"- Notable chains/versions: see left chart\n"
        f"- Top pools: see top-10 chart\n"
        f"What to watch next: monitor unusual deltas vs prior 24h."
    )

use_llm = st.sidebar.checkbox("Use OpenAI (if key set)", value=True)
if use_llm and "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        metrics = {
            "as_of_utc": as_of,
            "totals": {"volume_usd": kpi_volume, "swaps": kpi_swaps, "unique_traders": kpi_traders},
            "by_chain_version": (
                view.groupby([chain_col, version_col])[vol_col].sum().reset_index().to_dict(orient="records")
                if chain_col and version_col and vol_col else []
            ),
            "top_pools": (
                view.groupby(pool_label)[vol_col].sum().reset_index().sort_values(vol_col, ascending=False).head(5).to_dict(orient="records")
                if pool_label and vol_col else []
            ),
        }
        prompt = f"""
You are a product-minded analyst. Using ONLY the JSON below, write a Uniswap 24h digest:
- Headline (<= 15 words)
- 3 bullets: (1) overall numbers/trends, (2) notable chains/versions, (3) top pools/movers
- One “What to watch next” line
Use units (USD, %, 24h). Do not invent numbers.
JSON:
{json.dumps(metrics)}
"""
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        st.code(resp.output_text, language="markdown")
    except Exception as e:
        st.warning(f"LLM disabled (error: {e}). Showing fallback.")
        st.code(fallback_digest(), language="markdown")
else:
    st.info("OpenAI key not set. Showing fallback.")
    st.code(fallback_digest(), language="markdown")

st.divider()
with st.expander("Raw table (filtered)"):
    st.dataframe(view, use_container_width=True)
