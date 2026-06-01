import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Dashboard EDA Emosi — Marcus Aurelius & Ataraxia",
    page_icon="🧠",
    layout="wide"
)

DATA_PATH = "dataset_combined_cleaned.csv"

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

EMOTION_COLORS = {
    "netral": "#94a3b8",
    "cape":   "#f59e0b",
    "jijik":  "#84cc16",
    "marah":  "#ef4444",
    "senang": "#f472b6",
    "kaget":  "#a78bfa",
    "sedih":  "#60a5fa",
    "takut":  "#fb923c",
}

EMOTION_EMOJI = {
    "netral": "😐",
    "cape":   "😩",
    "jijik":  "🤢",
    "marah":  "😡",
    "senang": "😄",
    "kaget":  "😱",
    "sedih":  "😢",
    "takut":  "😨",
}

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        st.stop()

    df.columns = df.columns.astype(str).str.strip().str.lower()

    required_cols = ["label", "text"]
    for col in required_cols:
        if col not in df.columns:
            st.error(
                f"Kolom '{col}' tidak ditemukan.\n\n"
                f"Kolom yang tersedia: {list(df.columns)}"
            )
            st.stop()

    df = df.dropna(subset=["label", "text"])
    df["label"]      = df["label"].astype(str).str.strip().str.lower()
    df["text"]       = df["text"].astype(str).str.strip()
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["char_length"] = df["text"].apply(lambda x: len(x))

    return df


df = load_data()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🧠 EDA Emosi")
st.sidebar.caption("Dataset Gabungan — Marcus Aurelius & Ataraxia")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "📋 Overview",
        "📊 Distribusi Emosi",
        "📝 Panjang Teks",
        "☁️ Kata Dominan",
        "📌 Kesimpulan",
        "🛠 Debug Dataset"
    ]
)

# ==================================================
# OVERVIEW
# ==================================================
if menu == "📋 Overview":

    st.title("🧠 Dashboard EDA Emosi")
    st.caption("Dataset Gabungan: Marcus Aurelius & Ataraxia — 8 Kelas Emosi")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Emosi", df["label"].nunique())
    with col3:
        st.metric("Rata-rata Kata", round(df["word_count"].mean(), 1))
    with col4:
        st.metric("Rata-rata Karakter", round(df["char_length"].mean(), 1))

    st.divider()
    st.subheader("8 Kelas Emosi")

    cols = st.columns(8)
    for i, emo in enumerate(EXPECTED_EMOTIONS):
        count = len(df[df["label"] == emo])
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    background:{EMOTION_COLORS[emo]}22;
                    border:1.5px solid {EMOTION_COLORS[emo]};
                    border-radius:12px;
                    padding:12px 8px;
                    text-align:center;
                ">
                    <div style="font-size:2rem">{EMOTION_EMOJI[emo]}</div>
                    <div style="font-weight:700;font-size:0.95rem">{emo}</div>
                    <div style="font-size:0.8rem;color:#aaa">{count:,} data</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader("🔍 Preview Dataset")
    st.dataframe(
        df[["label", "text", "word_count", "char_length"]].head(10),
        use_container_width=True
    )

# ==================================================
# DISTRIBUSI EMOSI
# ==================================================
elif menu == "📊 Distribusi Emosi":

    st.title("📊 Distribusi Emosi")

    emotion_counts = (
        df["label"]
        .value_counts()
        .reindex(EXPECTED_EMOTIONS, fill_value=0)
    )

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        bars = ax.bar(
            [f"{EMOTION_EMOJI[e]} {e}" for e in emotion_counts.index],
            emotion_counts.values,
            color=[EMOTION_COLORS[e] for e in emotion_counts.index],
            edgecolor="none",
            width=0.65
        )

        for bar, val in zip(bars, emotion_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{val:,}",
                ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold"
            )

        ax.set_title("Jumlah Data per Emosi", color="white", fontsize=14, pad=15)
        ax.set_xlabel("Emosi", color="#94a3b8")
        ax.set_ylabel("Jumlah Data", color="#94a3b8")
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.yaxis.grid(True, color="#1e293b", linewidth=0.8)
        ax.set_axisbelow(True)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    with col_table:
        st.subheader("Tabel Distribusi")
        result_df = pd.DataFrame({
            "Emosi":  [f"{EMOTION_EMOJI[e]} {e}" for e in emotion_counts.index],
            "Jumlah": emotion_counts.values,
            "Persen": [f"{v / len(df) * 100:.1f}%" for v in emotion_counts.values]
        })
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor("#0f172a")
        ax2.pie(
            emotion_counts.values,
            labels=emotion_counts.index,
            colors=[EMOTION_COLORS[e] for e in emotion_counts.index],
            autopct="%1.1f%%",
            textprops={"color": "white", "fontsize": 8},
            startangle=90,
            wedgeprops={"linewidth": 2, "edgecolor": "#0f172a"}
        )
        ax2.set_title("Proporsi Emosi", color="white", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)

# ==================================================
# PANJANG TEKS
# ==================================================
elif menu == "📝 Panjang Teks":

    st.title("📝 Analisis Panjang Teks")

    tab1, tab2 = st.tabs(["Jumlah Kata", "Jumlah Karakter"])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        sns.boxplot(
            data=df,
            x="label",
            y="word_count",
            order=EXPECTED_EMOTIONS,
            palette=EMOTION_COLORS,
            ax=ax,
            linewidth=1.2,
            flierprops={"markerfacecolor": "#475569", "markersize": 3}
        )
        ax.set_title("Distribusi Jumlah Kata per Emosi", color="white", fontsize=13, pad=12)
        ax.set_xlabel("Emosi", color="#94a3b8")
        ax.set_ylabel("Jumlah Kata", color="#94a3b8")
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.yaxis.grid(True, color="#1e293b", linewidth=0.8)
        ax.set_axisbelow(True)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Statistik Jumlah Kata")
        word_stats = df.groupby("label")["word_count"].describe().round(2)
        st.dataframe(word_stats.reindex(EXPECTED_EMOTIONS), use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        sns.boxplot(
            data=df,
            x="label",
            y="char_length",
            order=EXPECTED_EMOTIONS,
            palette=EMOTION_COLORS,
            ax=ax,
            linewidth=1.2,
            flierprops={"markerfacecolor": "#475569", "markersize": 3}
        )
        ax.set_title("Distribusi Panjang Karakter per Emosi", color="white", fontsize=13, pad=12)
        ax.set_xlabel("Emosi", color="#94a3b8")
        ax.set_ylabel("Jumlah Karakter", color="#94a3b8")
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.yaxis.grid(True, color="#1e293b", linewidth=0.8)
        ax.set_axisbelow(True)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Statistik Jumlah Karakter")
        char_stats = df.groupby("label")["char_length"].describe().round(2)
        st.dataframe(char_stats.reindex(EXPECTED_EMOTIONS), use_container_width=True)

# ==================================================
# WORDCLOUD
# ==================================================
elif menu == "☁️ Kata Dominan":

    st.title("☁️ Kata Dominan per Emosi")

    STOPWORDS_ID = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "saya", "aku",
        "kamu", "dia", "kami", "mereka", "adalah", "ada", "dengan",
        "untuk", "tidak", "bisa", "akan", "sudah", "juga", "karena",
        "dalam", "pada", "oleh", "atau", "kata", "pesan", "marcus",
        "aurelius", "meditations", "seolah", "olah", "pura", "selalu", "sangat"
    }

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_emotion = st.selectbox(
            "Pilih Emosi",
            [f"{EMOTION_EMOJI[e]} {e}" for e in EXPECTED_EMOTIONS]
        )
        show_all = st.checkbox("Tampilkan semua emosi sekaligus", value=False)

    emo_key = selected_emotion.split(" ", 1)[1]

    if show_all:
        st.subheader("WordCloud — Semua Emosi")
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        fig.patch.set_facecolor("#0f172a")
        for ax, emo in zip(axes.flatten(), EXPECTED_EMOTIONS):
            txt = " ".join(df[df["label"] == emo]["text"])
            if txt.strip():
                wc = WordCloud(
                    width=400, height=250,
                    background_color=None,
                    mode="RGBA",
                    color_func=lambda *args, _emo=emo, **kwargs: EMOTION_COLORS[_emo],
                    max_words=60
                ).generate(txt)
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{EMOTION_EMOJI[emo]} {emo}", color="white", fontsize=11, pad=6)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        text = " ".join(df[df["label"] == emo_key]["text"])
        if text.strip():
            color_hex = EMOTION_COLORS[emo_key]

            wordcloud = WordCloud(
                width=1200, height=500,
                background_color=None,
                mode="RGBA",
                color_func=lambda *args, **kwargs: color_hex,
                max_words=100,
                prefer_horizontal=0.85
            ).generate(text)

            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(
                f"{EMOTION_EMOJI[emo_key]} {emo_key.upper()} — Kata Dominan",
                color="white", fontsize=14, pad=12
            )
            plt.tight_layout()
            st.pyplot(fig)

            # Top 10 kata
            words    = text.lower().split()
            filtered = [w for w in words if w not in STOPWORDS_ID and len(w) > 2]
            top_words = Counter(filtered).most_common(10)

            st.subheader("🔝 Top 10 Kata")
            cols_w = st.columns(5)
            for i, (word, freq) in enumerate(top_words):
                with cols_w[i % 5]:
                    st.markdown(
                        f"""<div style="
                            background:{color_hex}22;
                            border:1px solid {color_hex};
                            border-radius:8px;
                            padding:8px;
                            text-align:center;
                            margin-bottom:8px;
                        ">
                            <b>{word}</b><br>
                            <small>{freq}x</small>
                        </div>""",
                        unsafe_allow_html=True
                    )
        else:
            st.warning(f"Tidak ada data untuk emosi '{emo_key}'")

# ==================================================
# KESIMPULAN
# ==================================================
elif menu == "📌 Kesimpulan":

    st.title("📌 Kesimpulan")

    dominant_emotion = df["label"].value_counts().idxmax()
    minority_emotion = df["label"].value_counts().idxmin()

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            f"✅ Emosi paling banyak: **{EMOTION_EMOJI[dominant_emotion]} {dominant_emotion}** "
            f"({df['label'].value_counts()[dominant_emotion]:,} data)"
        )
    with col2:
        st.warning(
            f"⚠️ Emosi paling sedikit: **{EMOTION_EMOJI[minority_emotion]} {minority_emotion}** "
            f"({df['label'].value_counts()[minority_emotion]:,} data)"
        )

    st.divider()

    st.write(f"""
### Ringkasan Dataset Gabungan

Dataset ini merupakan gabungan dari dua sumber:
- **Marcus Aurelius**: kutipan-kutipan filosofis yang dianotasi dengan emosi
- **Ataraxia**: kalimat ekspresi sehari-hari dalam Bahasa Indonesia

| Metrik | Nilai |
|--------|-------|
| Total data | **{len(df):,}** |
| Jumlah kelas emosi | **{df['label'].nunique()}** |
| Rata-rata jumlah kata | **{df['word_count'].mean():.1f}** |
| Rata-rata jumlah karakter | **{df['char_length'].mean():.1f}** |
| Teks terpendek | **{df['word_count'].min()} kata** |
| Teks terpanjang | **{df['word_count'].max()} kata** |

### Distribusi Kelas
Dataset relatif **seimbang** antar kelas emosi ({df['label'].value_counts().min():,}–{df['label'].value_counts().max():,} data per kelas),
sehingga cocok untuk training model klasifikasi teks tanpa perlu oversampling berlebihan.

### Karakteristik Teks
- Teks dari **Marcus Aurelius** cenderung lebih panjang dan filosofis
- Teks dari **Ataraxia** cenderung lebih pendek dan langsung
- Keduanya berbahasa **Indonesia**
    """)

# ==================================================
# DEBUG
# ==================================================
elif menu == "🛠 Debug Dataset":

    st.title("🛠 Debug Dataset")

    st.subheader("Jumlah Data")
    st.write(f"{len(df):,}")

    st.subheader("Jumlah Emosi")
    st.write(df["label"].nunique())

    st.subheader("Daftar Emosi")
    st.write(sorted(df["label"].unique()))

    st.subheader("Distribusi Label")
    dist_df = df["label"].value_counts().reset_index()
    dist_df.columns = ["Emosi", "Jumlah"]
    st.dataframe(dist_df, use_container_width=True)

    st.subheader("Statistik Kolom Numerik")
    st.dataframe(
        df[["word_count", "char_length"]].describe().round(2),
        use_container_width=True
    )

    st.subheader("Cek Null")
    st.dataframe(df.isnull().sum().rename("Null Count"), use_container_width=True)

    st.subheader("Representasi Asli Label")
    for item in sorted(df["label"].unique()):
        st.code(repr(item))

    st.subheader("Sample Data (10 baris acak)")
    st.dataframe(
        df.sample(10)[["label", "text", "word_count", "char_length"]],
        use_container_width=True
    )