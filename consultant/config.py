"""Central configuration: paths, model ids, retrieval parameters."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Paths -----------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = ROOT / "eval"

CLINICAL_DIR = ROOT / "Clinical"
FOOD_DIR = ROOT / "FoodSecurity"
# User-supplied SEC filings: one sub-folder per company containing 10-K/10-Q PDFs.
SEC_LOCAL_DIR = ROOT / "UnitedStatesSecuritiesandExchangeCommission(SEC)"
RETAIL_CSV = ROOT / "Retail" / "CMPE256_Hackathon_market_basket_analysis_Release.csv"
CISCO_SOURCES = ROOT / "DataSource-Cisco.txt"
AGILENT_SOURCES = ROOT / "Agilent.txt"
GCP_SOURCES = ROOT / "Google GCP.txt"

# --- Models ----------------------------------------------------------------
# Domain agents, critic and synthesis. Haiku keeps costs low; set AGENT_MODEL=claude-sonnet-5-5 for higher quality.
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-haiku-5-5")
# Routing, catalog extraction and LLM-as-judge.
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-5-5")
# Server-side refusal fallback (Claude API only; set to 0 on Bedrock/Vertex).
USE_FALLBACKS = os.getenv("USE_FALLBACKS", "1") == "1"

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

# --- Retrieval -------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_TOKENS = 450          # approx words*1.3; bge-small max is 512 tokens
CHUNK_OVERLAP = 60
CANDIDATES_PER_RETRIEVER = 30
RRF_K = 60
DEFAULT_TOP_K = 6

# ClinicalTrials.gov exports are large (up to ~80k rows); cap per condition.
CLINICAL_TRIALS_PER_CONDITION = int(os.getenv("CLINICAL_TRIALS_PER_CONDITION", "2500"))

# --- Orchestration ---------------------------------------------------------
MAX_AGENT_TOOL_TURNS = 8
MAX_GOAL_LOOPS = 2

DOMAINS = ["technology", "clinical", "sec", "csr", "procurement"]
DOMAIN_LABELS = {
    "technology": "Technology",
    "clinical": "Clinical",
    "sec": "SEC",
    "csr": "Corporate Responsibility",
    "procurement": "Procurement",
}
