import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import re

st.set_page_config(
    page_title="EDA Emosi Markus Erelius",
    page_icon="📊",
    layout="wide",
)

# =========================
# KONFIGURASI
# =========================
DATA_PATH = Path("data/dataset_markus_erelius_clean.csv")

COLORS = {
    "netral": "#6B7280",
    "senang": "#F59E0B",
    "sedih": "#3B82F6",
    "marah": "#EF4444",
    "takut": "#8B5CF6",
    "jijik": "#10B981",
    "kaget": "#F97316",
    "cape": "#6366F1",
}

ORDER = [
    "netral",
    "senang",
    "sedih",
    "marah",
    "takut",
    "jijik",
    "kaget",
    "cape",
]

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            f"""
File tidak ditemukan:

{DATA_PATH}

Pastikan struktur repository:

project/
│
├── app.py
├── requirements.txt
└── data/
    └── dataset_markus_erelius_clean.csv
"""
        )
        st.stop()

    try:
        df = pd.read_csv(
            DATA_PATH,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip"
        )

    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()

    required_cols = ["label", "text"]

    for col in required_cols:
        if col not in df.columns:
            st.error(
                f"Kolom '{col}' tidak ditemukan.\n\n"
                f"Kolom yang tersedia: {list(df.columns)}"
            )
            st.stop()

    df = df.dropna(subset=["label", "text"])

    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str)

    df["word_count"] = df["text"].str.split().str.len()
    df["char_length"] = df["text"].str.len()

    return df


df = load_data()

st.success(f"Dataset berhasil dimuat ({len(df)} baris)")