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

# ---------- Config (set in Streamlit Secrets or env) ----------
WINDOW_DEFAULT = int(os.getenv("WINDOW_HOURS", "24"))  # affects Q_VOLUME only
Q_POOLS_ID    = int(os.getenv("DUNE_Q_POOLS_ID", "0"))     # your Query 1 (top pools, fixed 24h)
Q_ACTIVITY_ID = int(os.getenv("DUNE_Q_ACTIVITY_ID", "0"))  # your Query 2 (swaps & traders, fixed 24h)
Q_VOLUME_ID   = int(os.getenv("DUNE_Q_VOLUME_ID", "0"))    # your Query 3 (volume by chain/version, param)

# ---------- Sidebar ----------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Choose source", ["Dune API", "Local JSON (debug)"], horizontal=True)
window_hours = st.sidebar.number_input("Window (hours, affects Volume query only)", 1, 168, WINDOW_DEFAULT)
json_pools    = st.sidebar.text_input("Local JSON for Top Pools (Q1)", value="q1_pools.json")
json_activity = st.sidebar.text_input("Local JSON for Activity (Q2)", value="q2_activity.json")
json_volume   = st.sidebar.text_input("Local JSON for Volume (Q3)", value="q3_volume.json")

st.caption("Note: Q1 & Q2 are hard-coded 24h in SQL. `window_hours` controls Q3 only unless you edit those queries.")

# ---------- Helpers ----------
def _safe_float(x):
    try: return float(x)
    except Exception: return None

def fmt_usd(x):
    try:
        x = float(x)
        if x >= 1e9:  return f"${x/1e9:.2f}B"
        if x >= 1e6:  return f"${x/1e6:.2f}M"
        if x >= 1e3:  return f"${x/1e3:.2f}K"
        return f"${x:,.0f}"
    except Exception:
        return "—"

# ---------- Loaders ----------
@st.cache_data(ttl=15*60)
def dune_df_no_param(query_id: int) -> pd.DataFrame:
    """Run a Dune query with NO params (for your Q1/Q2 fixed-24h SQL)."""
    if not HAVE_DUNE:
        raise RuntimeError("Dune SDK not available")
    dune = DuneClient(api_key=st.secrets["DUNE_API_KEY"])
    return dune.run_query_dataframe(QueryBase(query_id=query_id))

@st.cache_data(ttl=15*60)
def dune_df_with_window(query_id: int, window_hours: int) -> pd.DataFrame:
    """Run a Dune query with window_hours param (INTERVAL '{{window_hours}}' hour), else fallback to no-param."""
    if not HAVE_DUNE:
        raise RuntimeError("Dune SDK not available")
    dune = DuneClient(api_key=st.secrets["DUNE_API_KEY"])
    try:
        q = QueryBase(query_id=query_id, params=[QueryParameter.number_type("window_hours", int(window_hours))])
        return dune.run_query_dataframe(q)
    except Exception as e:
        # If param isn't defined in SQL, retry no-param for resilience
        msg = str(e).lower()
        if "unknown parameter" in msg or "unknown parameters" in msg:
            return dune.run_query_dataframe(QueryBase(query_id=query_id))
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

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase-insensitive aliasing to canonical names across queries."""
    if df.empty: return df
    out = df.copy()
    low = {c.lower(): c for c in out.columns}
    def alias(want, *cands):
        for c in cands:
            if c in low and low[c] != want:
                out.rename(columns={low[c]: want}, inplace=True)
                return
        if want in low:  # already correct name
            return
    alias("chain", "chain", "blockchain")
    alias("version", "version", "dex_version", "pool_version")
    alias("volume_usd", "volume_usd", "usd_amount", "amount_usd", "usd_volume", "sum_amount_usd")
    alias("pool_or_pair", "pool_or_pair", "pool", "token_pair")
    alias("token_symbol", "token_symbol", "token_bought_symbol")
    alias("swaps", "swaps", "swap_count")
    alias("unique_traders", "unique_traders", "traders", "unique_addresses")
    alias("unique_txs", "unique_txs")
    alias("tx_hash", "tx_hash", "hash", "txhash")
    alias("tx_from", "tx_from", "trader", "from_address")
    return out

# ---------- Load data ----------
as_of = datetime.now(timezone.utc).isoformat()

if mode == "Dune API":
    if "DUNE_API_KEY" not in st.secrets or not st.secrets["DUNE_API_KEY"]:
        st.error("Add DUNE_API_KEY in Streamlit Secrets.")
        st.stop()
    try:
        df_pools    = dune_df_no_param(Q_POOLS_ID)     if Q_POOLS_ID    else pd.DataFrame()
        df_activity = dune_df_no_param(Q_ACTIVITY_ID)  if Q_ACTIVITY_ID else pd.DataFrame()
        df_volume   = dune_df_with_window(Q_VOLUME_ID, window_hours) if Q_VOLUME_ID else pd.DataFrame()
    except Exception as e:
        st.error(f"Dune API error: {e}")
        st.stop()
else:
    df_pools    = load_json(json_pools)    if os.path.exists(json_pools)    else pd.DataFrame()
    df_activity = load_json(json_activity) if os.path.exists(json_activity) else pd.DataFrame()
    df_volume   = load_json(json_volume)   if os.path.exists(json_volume)   else pd.DataFrame()

df_pools, df_activity, df_volume = normalize(df_pools), normalize(df_activity), normalize(df_volume)

if df_pools.empty and df_activity.empty and df_volume.empty:
    st.warning("No data returned. Provide at least one query/JSON.")
    st.stop()

# ---------- Build unified metrics ----------
# Volumes (from Q3)
total_volume, by_chain_version, version_share = None, [], []
if not df_volume.empty and "volume_usd" in df_volume.columns:
    total_volume = _safe_float(df_volume["volume_usd"].sum())
    if "chain" in df_volume.columns and "version" in df_volume.columns:
        by_chain_version = (
            df_volume.groupby(["chain","version"])["volume_usd"]
                     .sum().reset_index()
                     .sort_values(["chain","version"])
                     .to_dict(orient="records")
        )
    if "version" in df_volume.columns:
        by_ver = (
            df_volume.groupby("version")["volume_usd"]
                     .sum().reset_index()
                     .sort_values("volume_usd", ascending=False)
        )
        total = by_ver["volume_usd"].sum()
        if total > 0:
            by_ver["share_pct"] = (by_ver["volume_usd"] / total * 100).round(2)
            version_share = by_ver.to_dict(orient="records")

# Activity (from Q2)
total_swaps, total_traders = None, None
activity_by_chain_version = []
if not df_activity.empty:
    # Summaries
    if "swaps" in df_activity.columns:
        total_swaps = int(pd.to_numeric(df_activity["swaps"], errors="coerce").fillna(0).sum())
    elif "tx_hash" in df_activity.columns:
        total_swaps = int(df_activity["tx_hash"].nunique())
    if "unique_traders" in df_activity.columns:
        total_traders = int(pd.to_numeric(df_activity["unique_traders"], errors="coerce").fillna(0).sum())
    elif "tx_from" in df_activity.columns:
        total_traders = int(df_activity["tx_from"].nunique())
    # By chain/version (if present)
    has_chain = "chain" in df_activity.columns
    has_ver   = "version" in df_activity.columns
    if has_chain and has_ver:
        cols = [c for c in ["swaps","unique_traders"] if c in df_activity.columns]
        if cols:
            gb = df_activity.groupby(["chain","version"])[cols].sum().reset_index()
            activity_by_chain_version = gb.to_dict(orient="records")

# Top pools (from Q1)
top_pools = []
if not df_pools.empty:
    # Expect: chain, version, pool_or_pair (or pool/token_symbol), volume_usd
    # Normalize a display label
    if "pool_or_pair" not in df_pools.columns:
        if "token_symbol" in df_pools.columns:
            df_pools["pool_or_pair"] = df_pools["token_symbol"]
        elif "pool" in df_pools.columns:
            df_pools["pool_or_pair"] = df_pools["pool"]
    if "volume_usd" in df_pools.columns and "pool_or_pair" in df_pools.columns:
        tp = (
            df_pools.groupby("pool_or_pair")["volume_usd"].sum()
                    .reset_index()
                    .sort_values("volume_usd", ascending=False)
                    .head(10)
        )
        top_pools = tp.to_dict(orient="records")

# ---------- LLM: Executive digest (first) ----------
st.subheader("Daily Digest (LLM)")

def payload_for_llm():
    p = {
        "as_of_utc": as_of,
        "window_hours": int(window_hours),  # applies to Q3
        "totals": {}
    }
    if total_volume is not None:  p["totals"]["volume_usd"] = round(total_volume, 2)
    if total_swaps  is not None:  p["totals"]["swaps"] = int(total_swaps)
    if total_traders is not None: p["totals"]["unique_traders"] = int(total_traders)
    if by_chain_version:          p["by_chain_version"] = by_chain_version
    if activity_by_chain_version: p["activity_by_chain_version"] = activity_by_chain_version
    if version_share:             p["version_share"] = version_share
    if top_pools:                 p["top_pools_by_volume"] = top_pools
    return p

def render_digest_md(d):
    headline = d.get("headline", f"Uniswap activity (last {int(window_hours)}h)")
    bullets  = d.get("bullets", [])
    watch    = d.get("watch_next", "")
    md = f"### {headline}\n\n"
    for b in bullets:
        md += f"- {b}\n"
    if watch:
        md += f"\n**What to watch next:** {watch}\n"
    return md

def fallback_digest_md():
    parts = [f"### Uniswap activity (last {int(window_hours)}h)"]
    line = []
    if total_volume is not None:  line.append(f"Volume: {fmt_usd(total_volume)}")
    if total_swaps  is not None:  line.append(f"Swaps: {total_swaps:,}")
    if total_traders is not None: line.append(f"Unique traders: {total_traders:,}")
    if version_share:
        try:
            v4 = next((r for r in version_share if str(r.get("version")) == "4"), None)
            if v4: line.append(f"v4 share: {v4['share_pct']:.2f}%")
        except Exception:
            pass
    if line: parts.append("- " + " • ".join(line))
    if by_chain_version: parts.append("- See chain/version breakdown below.")
    if top_pools: parts.append("- Top pools included in analysis.")
    parts.append("**What to watch next:** track unusual deltas by chain/version and large pool rotations.")
    return "\n".join(parts)

use_llm = st.sidebar.checkbox("Use OpenAI (if key set)", value=True)
digest_md = fallback_digest_md()

if use_llm and "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        payload = payload_for_llm()

        system = (
            "You are a product-minded analyst for Uniswap. "
            "Return ONLY JSON with keys: headline (string), bullets (array of 3-4 strings), "
            "watch_next (string, optional). Use ONLY provided metrics. "
            "If version_share is present, consider calling out v4 vs others. "
            "If top_pools_by_volume is present, mention notable tokens/pools. "
            "Be concise, executive-friendly, and numeric where it helps."
        )
        user = f"Metrics JSON:\n{json.dumps(payload)}"

        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type":"json_object"}
        )
        raw = comp.choices[0].message.content
        dig = json.loads(raw) if raw else {}
        if isinstance(dig, dict):
            digest_md = render_digest_md(dig)
    except Exception as e:
        st.warning(f"LLM disabled (error: {e}). Showing fallback.")

st.markdown(digest_md)

# ---------- KPI strip ----------
st.markdown("---")
kpis = []
if total_volume is not None:  kpis.append(("Volume (USD, window)", fmt_usd(total_volume)))
if total_swaps  is not None:  kpis.append(("Swaps (window)", f"{total_swaps:,}"))
if total_traders is not None: kpis.append(("Unique traders (window)", f"{total_traders:,}"))
if kpis:
    cols = st.columns(len(kpis))
    for c, (label, value) in zip(cols, kpis):
        c.metric(label, value)
st.caption(f"As of (UTC): {as_of} • Window: last {int(window_hours)}h (affects Volume query only)")

# ---------- Chart: Volume by Chain & Version (from Q3) ----------
st.subheader("Volume by Chain & Version")
if by_chain_version:
    pivot = (
        pd.DataFrame(by_chain_version)
        .pivot(index="chain", columns="version", values="volume_usd")
        .fillna(0)
        .sort_index()
    )
    st.bar_chart(pivot)
else:
    st.info("Q3 needs columns: chain, version, volume_usd (or alias them).")

# ---------- Debug tables (optional) ----------
with st.expander("Debug: Top pools (Q1)"):
    st.dataframe(df_pools, use_container_width=True)
with st.expander("Debug: Activity (Q2)"):
    st.dataframe(df_activity, use_container_width=True)
with st.expander("Debug: Volume (Q3)"):
    st.dataframe(df_volume, use_container_width=True)
