import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path

# ==================================
# CONFIG
# ==================================
st.set_page_config(
    page_title="Dashboard EDA Emosi Marcus Aurelius",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "dataset_markus_erelius_clean.csv"

# ==================================
# LOAD DATA
# ==================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip"
    )

    # Bersihkan nama kolom
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(";", "", regex=False)
    )

    # Rename jika masih text;
    if "text;" in df.columns:
        df.rename(columns={"text;": "text"}, inplace=True)

    required_cols = {"label", "text"}

    if not required_cols.issubset(df.columns):
        st.error(f"Kolom ditemukan: {list(df.columns)}")
        st.stop()

    df = df.dropna(subset=["label", "text"])

    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str)

    # Feature Engineering
    df["word_count"] = df["text"].str.split().str.len()
    df["char_length"] = df["text"].str.len()

    return df


df = load_data()

# ==================================
# SIDEBAR
# ==================================
st.sidebar.title("📚 Dashboard EDA")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview",
        "Distribusi Emosi",
        "Panjang Teks",
        "Kata Dominan",
        "Kesimpulan"
    ]
)

# ==================================
# OVERVIEW
# ==================================
if menu == "Overview":

    st.title("📊 Dashboard Analisis Emosi Marcus Aurelius")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", len(df))
    col2.metric("Jumlah Emosi", df["label"].nunique())
    col3.metric("Rata-rata Kata", round(df["word_count"].mean(), 1))

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    st.subheader("Data Dictionary")

    dictionary = pd.DataFrame({
        "Kolom": [
            "label",
            "text",
            "word_count",
            "char_length"
        ],
        "Deskripsi": [
            "Kategori emosi",
            "Isi kutipan",
            "Jumlah kata",
            "Jumlah karakter"
        ]
    })

    st.dataframe(dictionary)

# ==================================
# DISTRIBUSI EMOSI
# ==================================
elif menu == "Distribusi Emosi":

    st.title("📈 Distribusi Emosi")

    emotion_counts = df["label"].value_counts()

    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(
        x=emotion_counts.index,
        y=emotion_counts.values,
        ax=ax
    )

    ax.set_title("Jumlah Data per Emosi")
    ax.set_xlabel("Emosi")
    ax.set_ylabel("Jumlah")

    st.pyplot(fig)

    st.dataframe(
        emotion_counts.reset_index().rename(
            columns={
                "index":"Emosi",
                "label":"Jumlah"
            }
        )
    )

# ==================================
# PANJANG TEKS
# ==================================
elif menu == "Panjang Teks":

    st.title("📝 Analisis Panjang Teks")

    st.subheader("Word Count")

    fig, ax = plt.subplots(figsize=(12,5))

    sns.boxplot(
        data=df,
        x="label",
        y="word_count",
        ax=ax
    )

    ax.set_title("Distribusi Jumlah Kata per Emosi")

    st.pyplot(fig)

    st.subheader("Character Length")

    fig, ax = plt.subplots(figsize=(12,5))

    sns.boxplot(
        data=df,
        x="label",
        y="char_length",
        ax=ax
    )

    ax.set_title("Distribusi Jumlah Karakter per Emosi")

    st.pyplot(fig)

# ==================================
# KATA DOMINAN
# ==================================
elif menu == "Kata Dominan":

    st.title("☁️ Kata Dominan per Emosi")

    emotions = sorted(df["label"].unique())

    selected_emotion = st.selectbox(
        "Pilih Emosi",
        emotions
    )

    text = " ".join(
        df[df["label"] == selected_emotion]["text"]
    )

    if len(text) > 0:

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12,6))

        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)

# ==================================
# KESIMPULAN
# ==================================
elif menu == "Kesimpulan":

    st.title("📌 Kesimpulan")

    dominant_emotion = (
        df["label"]
        .value_counts()
        .idxmax()
    )

    st.write(
        f"""
        ### Insight Utama

        - Dataset memiliki **{len(df)}** data.
        - Terdapat **{df['label'].nunique()}** kategori emosi.
        - Emosi yang paling dominan adalah **{dominant_emotion}**.
        - Panjang teks dapat dibandingkan menggunakan boxplot.
        - Wordcloud menunjukkan kata-kata yang sering muncul pada setiap emosi.
        """
    )