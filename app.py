import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ==================================
# CONFIG
# ==================================
st.set_page_config(
    page_title="Dashboard EDA Emosi Marcus Aurelius",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "dataset_markus_erelius_clean.csv"

EXPECTED_EMOTIONS = [
    "netral",
    "cape",
    "jijik",
    "marah",
    "senang",
    "kaget",
    "sedih",
    "takut"
]

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

    if "text;" in df.columns:
        df.rename(columns={"text;": "text"}, inplace=True)

    required_cols = {"label", "text"}

    if not required_cols.issubset(df.columns):
        st.error(f"Kolom ditemukan: {list(df.columns)}")
        st.stop()

    df = df.dropna(subset=["label", "text"])

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

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

    st.subheader("Daftar Emosi")

    emotion_df = pd.DataFrame({
        "Emosi": EXPECTED_EMOTIONS
    })

    st.dataframe(emotion_df)

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
            "Isi kutipan Marcus Aurelius",
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

    emotion_counts = (
        df["label"]
        .value_counts()
        .reindex(EXPECTED_EMOTIONS, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=emotion_counts.index,
        y=emotion_counts.values,
        ax=ax
    )

    ax.set_title("Jumlah Data per Emosi")
    ax.set_xlabel("Emosi")
    ax.set_ylabel("Jumlah")

    plt.xticks(rotation=20)

    st.pyplot(fig)

    summary_df = pd.DataFrame({
        "Emosi": emotion_counts.index,
        "Jumlah": emotion_counts.values
    })

    st.dataframe(summary_df)

# ==================================
# PANJANG TEKS
# ==================================
elif menu == "Panjang Teks":

    st.title("📝 Analisis Panjang Teks")

    st.subheader("Distribusi Jumlah Kata")

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.boxplot(
        data=df,
        x="label",
        y="word_count",
        order=EXPECTED_EMOTIONS,
        ax=ax
    )

    ax.set_title("Distribusi Jumlah Kata per Emosi")

    st.pyplot(fig)

    st.subheader("Distribusi Jumlah Karakter")

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.boxplot(
        data=df,
        x="label",
        y="char_length",
        order=EXPECTED_EMOTIONS,
        ax=ax
    )

    ax.set_title("Distribusi Jumlah Karakter per Emosi")

    st.pyplot(fig)

# ==================================
# KATA DOMINAN
# ==================================
elif menu == "Kata Dominan":

    st.title("☁️ Kata Dominan per Emosi")

    selected_emotion = st.selectbox(
        "Pilih Emosi",
        EXPECTED_EMOTIONS
    )

    emotion_text = " ".join(
        df[df["label"] == selected_emotion]["text"]
    )

    if emotion_text:

        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white"
        ).generate(emotion_text)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)

    else:
        st.warning("Tidak ada data untuk emosi ini.")

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

    st.success(
        f"Emosi yang paling dominan adalah **{dominant_emotion}**."
    )

    st.markdown(f"""
### Insight Utama

- Dataset berisi **{len(df):,}** kutipan.
- Memiliki **{df['label'].nunique()} kategori emosi**.
- Terdapat **8 emosi utama**:
  - Netral
  - Cape
  - Jijik
  - Marah
  - Senang
  - Kaget
  - Sedih
  - Takut
- Emosi dominan: **{dominant_emotion}**
- Panjang teks dianalisis menggunakan boxplot.
- Kata dominan dianalisis menggunakan WordCloud.
""")