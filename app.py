import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="EDA Emosi Markus Erelius",
    page_icon="📊",
    layout="wide",
)

# =========================
# KONFIGURASI
# =========================
DATA_PATH = Path("data/dataset_markus_erelius_clean.csv")


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(f"File tidak ditemukan: {DATA_PATH}")
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

    required_cols = {"label", "text"}

    if not required_cols.issubset(df.columns):
        st.error(
            f"Kolom ditemukan: {list(df.columns)}\n\n"
            "CSV harus memiliki kolom 'label' dan 'text'."
        )
        st.stop()

    df = df.dropna(subset=["label", "text"])

    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str)

    df["word_count"] = df["text"].str.split().str.len()
    df["char_length"] = df["text"].str.len()

    return df


# =========================
# MAIN
# =========================
df = load_data()

st.success(f"Dataset berhasil dimuat ({len(df)} baris)")
st.dataframe(df.head())