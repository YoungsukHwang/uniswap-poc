# Uniswap On-Chain 24h Summarizer (POC)

A tiny Streamlit app that visualizes Uniswap activity (24h) and generates a plain-English daily digest.

## Deploy on Streamlit Cloud
1) Push this repo to GitHub.
2) Go to https://share.streamlit.io → New app → pick this repo & `streamlit_app.py`.
3) In the app settings → **Secrets**, add:

DUNE_API_KEY = "..."
DUNE_QUERY_ID = "123456"   # your Dune query id returning columns like chain, version, volume_usd, etc.
OPENAI_API_KEY = "..."     # optional; app falls back if missing
WINDOW_HOURS = "24"

4) Deploy. You’ll get a link like `https://your-app.streamlit.app`.

## Local run
pip install -r requirements.txt
streamlit run streamlit_app.py
