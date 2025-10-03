# streamlit_app.py
import os, json, pandas as pd, streamlit as st
from datetime import datetime, timezone

# ---------- Optional imports ----------
try:
    from dune_client.client import DuneClient
    from dune_client.query import QueryBase
    from dune_client.types import QueryParameter
    HAVE_DUNE = True
except Exception:
    HAVE_DUNE = False

st.set_page_config(page_title="Uniswap On-Chain Summarizer (POC)", layout="wide")
st.title("Uniswap On-Chain Summarizer (POC)")

# ---------- Config ----------
WINDOW_DEFAULT = int(os.getenv("WINDOW_HOURS", "24"))
Q_POOLS_ID    = int(os.getenv("DUNE_Q_POOLS_ID", "0"))
Q_ACTIVITY_ID = int(os.getenv("DUNE_Q_ACTIVITY_ID", "0"))
Q_VOLUME_ID   = int(os.getenv("DUNE_Q_VOLUME_ID", "0"))

# ---------- Sidebar ----------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Source", ["Dune API", "Local JSON"], horizontal=True)
window_hours = st.sidebar.number_input("Window (hours)", 1, 168, WINDOW_DEFAULT)
json_pools    = st.sidebar.text_input("Local JSON (Q1 pools)", value="q1_pools.json")
json_activity = st.sidebar.text_input("Local JSON (Q2 activity)", value="q2_activity.json")
json_volume   = st.sidebar.text_input("Local JSON (Q3 volume)", value="q3_volume.json")

# ---------- Helpers ----------
def _safe_float(x):
    try: return float(x)
    except: return None

def fmt_usd(x):
    try:
        x = float(x)
        if x >= 1e9: return f"${x/1e9:.2f}B"
        if x >= 1e6: return f"${x/1e6:.2f}M"
        if x >= 1e3: return f"${x/1e3:.0f}K"
        return f"${x:,.0f}"
    except: return "—"

@st.cache_data(ttl=15*60)
def dune_df(query_id: int, window_hours: int) -> pd.DataFrame:
    if not HAVE_DUNE:
        raise RuntimeError("Dune SDK not available")
    dune = DuneClient(api_key=st.secrets["DUNE_API_KEY"])
    q = QueryBase(
        query_id=query_id,
        params=[QueryParameter.number_type("window_hours", int(window_hours))]
    )
    return dune.run_query_dataframe(q)

@st.cache_data
def load_json(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict) and "rows" in data:
        return pd.DataFrame(data["rows"])
    return pd.DataFrame(data.get("rows", []))

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    low = {c.lower(): c for c in out.columns}
    def alias(want, *cands):
        for c in cands:
            if c in low: 
                out.rename(columns={low[c]: want}, inplace=True)
                return
    alias("chain", "chain", "blockchain")
    alias("version", "version")
    alias("volume_usd", "volume_usd", "amount_usd", "usd_amount")
    alias("pool_or_pair", "pool_or_pair", "pool", "token_pair")
    alias("token_symbol", "token_symbol", "token_bought_symbol")
    alias("swaps", "swaps")
    alias("unique_traders", "unique_traders", "trader_count")
    return out

# ---------- Load data ----------
as_of = datetime.now(timezone.utc).isoformat()

if mode == "Dune API":
    if "DUNE_API_KEY" not in st.secrets:
        st.error("Missing DUNE_API_KEY in Streamlit Secrets.")
        st.stop()
    try:
        df_pools    = dune_df(Q_POOLS_ID, window_hours)    if Q_POOLS_ID else pd.DataFrame()
        df_activity = dune_df(Q_ACTIVITY_ID, window_hours) if Q_ACTIVITY_ID else pd.DataFrame()
        df_volume   = dune_df(Q_VOLUME_ID, window_hours)   if Q_VOLUME_ID else pd.DataFrame()
    except Exception as e:
        st.error(f"Dune API error: {e}")
        st.stop()
else:
    df_pools    = load_json(json_pools)
    df_activity = load_json(json_activity)
    df_volume   = load_json(json_volume)

df_pools, df_activity, df_volume = map(normalize, [df_pools, df_activity, df_volume])

if df_pools.empty and df_activity.empty and df_volume.empty:
    st.warning("No data returned. Provide valid query IDs or JSONs.")
    st.stop()

# ---------- Metrics assembly ----------
total_volume = _safe_float(df_volume["volume_usd"].sum()) if "volume_usd" in df_volume else None
by_chain_version = (
    df_volume.groupby(["chain","version"])["volume_usd"].sum().reset_index().to_dict("records")
    if not df_volume.empty and "chain" in df_volume and "version" in df_volume else []
)

total_swaps = int(df_activity["swaps"].sum()) if "swaps" in df_activity else None
total_traders = int(df_activity["unique_traders"].sum()) if "unique_traders" in df_activity else None

top_pools = (
    df_pools.groupby("pool_or_pair")["volume_usd"].sum().reset_index().sort_values("volume_usd", ascending=False).head(10).to_dict("records")
    if not df_pools.empty and "pool_or_pair" in df_pools and "volume_usd" in df_pools else []
)

# ---------- LLM Digest ----------
st.subheader("Daily Digest (LLM)")

def fallback_digest():
    parts = [f"### Uniswap Summary (last {window_hours}h)"]
    line = []
    if total_volume: line.append(f"Volume {fmt_usd(total_volume)}")
    if total_swaps: line.append(f"Swaps {total_swaps:,}")
    if total_traders: line.append(f"Unique traders {total_traders:,}")
    if line: parts.append("- " + " • ".join(line))
    if top_pools: parts.append("- Top pools included (see debug table).")
    return "\n".join(parts)

digest_md = fallback_digest()

if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        payload = {
            "as_of_utc": as_of,
            "window_hours": window_hours,
            "totals": {
                "volume_usd": total_volume,
                "swaps": total_swaps,
                "unique_traders": total_traders,
            },
            "by_chain_version": by_chain_version,
            "top_pools": top_pools,
        }
        system = (
            "You are a product-minded analyst. "
            "Summarize Uniswap’s last {window_hours}h activity from provided JSON. "
            "Give headline + 3 concise bullets + 1 'what to watch next'."
        )
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":json.dumps(payload)}
            ],
            response_format={"type":"json_object"}
        )
        raw = comp.choices[0].message.content
        dig = json.loads(raw)
        headline = dig.get("headline", "")
        bullets  = dig.get("bullets", [])
        watch    = dig.get("watch_next", "")
        digest_md = f"### {headline}\n\n" + "\n".join([f"- {b}" for b in bullets])
        if watch: digest_md += f"\n\n**What to watch next:** {watch}"
    except Exception as e:
        st.warning(f"LLM disabled (error: {e}). Showing fallback.")

st.markdown(digest_md)

# ---------- KPIs ----------
st.markdown("---")
cols = []
if total_volume: cols.append(("Volume", fmt_usd(total_volume)))
if total_swaps: cols.append(("Swaps", f"{total_swaps:,}"))
if total_traders: cols.append(("Unique Traders", f"{total_traders:,}"))
if cols:
    c = st.columns(len(cols))
    for col, (label, val) in zip(c, cols):
        col.metric(label, val)
st.caption(f"As of (UTC): {as_of} • Window: {window_hours}h")

# ---------- Chart ----------
st.subheader("Volume by Chain & Version")
if by_chain_version:
    pivot = pd.DataFrame(by_chain_version).pivot(index="chain", columns="version", values="volume_usd").fillna(0)
    st.bar_chart(pivot)
else:
    st.info("No volume data to chart.")

# ---------- Debug ----------
with st.expander("Debug: Top Pools (Q1)"): st.dataframe(df_pools)
with st.expander("Debug: Activity (Q2)"): st.dataframe(df_activity)
with st.expander("Debug: Volume (Q3)"): st.dataframe(df_volume)
