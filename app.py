import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import re

# ── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Emosi Markus Erelius",
    page_icon="📊",
    layout="wide",
)

# ── Konstanta ────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/dataset_markus_erelius_clean.csv")

COLORS = {
    "netral": "#6B7280", "senang": "#F59E0B", "sedih": "#3B82F6",
    "marah":  "#EF4444", "takut":  "#8B5CF6", "jijik": "#10B981",
    "kaget":  "#F97316", "cape":   "#6366F1",
}
ORDER = ["netral", "senang", "sedih", "marah", "takut", "jijik", "kaget", "cape"]

STOPWORDS = {
    "yang","dan","di","ke","dari","ini","itu","adalah","dengan","untuk",
    "pada","tidak","akan","sudah","ada","juga","karena","apa","atau",
    "kamu","aku","saya","mereka","kita","dia","kata","marcus","aurelius",
    "pesan","seperti","setiap","dalam","jadi","semua","lagi","tentang",
    "begitu","bisa","lebih","sangat","bila","jika","tapi","namun","serta",
    "bahwa","antara","secara","sebuah","hanya","dapat","masih","pun",
}

RCPARAMS = {
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# ── Helper ───────────────────────────────────────────────────────────────────
def top_words(texts, n=8):
    words = []
    for t in texts:
        found = re.findall(r"\b[a-z]{3,}\b", str(t).lower())
        words.extend([w for w in found if w not in STOPWORDS])
    return Counter(words).most_common(n)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["word_count"]  = df["text"].str.split().str.len()
    df["char_length"] = df["text"].str.len()
    return df


# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 EDA — Dataset Emosi Markus Erelius")
st.markdown(
    "Exploratory Data Analysis untuk dataset klasifikasi emosi berbahasa Indonesia "
    "dari kutipan Marcus Aurelius."
)
st.divider()

# ── Load data dari folder data/ ──────────────────────────────────────────────
if not DATA_PATH.exists():
    st.error(
        f"File **{DATA_PATH}** tidak ditemukan.  \n\n"
        "Pastikan struktur folder sudah benar:\n"
        "```\n"
        "eda-markus-erelius/\n"
        "├── data/\n"
        "│   └── dataset_markus_erelius_clean.csv\n"
        "└── app.py\n"
        "```"
    )
    st.stop()

df = load_data(DATA_PATH)

# Validasi kolom
if not {"label", "text"}.issubset(df.columns):
    st.error("File CSV harus memiliki kolom **label** dan **text**.")
    st.stop()

# ── Sidebar — filter label ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filter")
    available_labels = [l for l in ORDER if l in df["label"].unique()]
    selected = st.multiselect(
        "Pilih label emosi",
        options=available_labels,
        default=available_labels,
    )
    if not selected:
        st.warning("Pilih minimal 1 label.")
        st.stop()

    st.divider()
    st.caption(f"Sumber data: `{DATA_PATH}`")
    st.caption(f"Total data: **{len(df):,}** baris")
    st.caption(f"Label dipilih: **{len(selected)}** dari {len(available_labels)}")

df_filtered   = df[df["label"].isin(selected)].copy()
order_filtered = [l for l in ORDER if l in selected]

# ── Ringkasan cepat ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Data",      f"{len(df_filtered):,}")
c2.metric("Label Aktif",     len(selected))
c3.metric("Rata-rata Kata",  f"{df_filtered['word_count'].mean():.1f}")
c4.metric("Teks Terpanjang", f"{df_filtered['word_count'].max()} kata")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Q1 — Distribusi kelas
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Q1 — Apakah jumlah data tiap emosi sudah cukup dan merata?")
st.markdown(
    "**Kenapa penting:** Model bisa bias ke kelas yang datanya jauh lebih banyak. "
    "Kalau tidak merata, perlu teknik balancing sebelum training."
)

counts = df_filtered["label"].value_counts().reindex(order_filtered).dropna()

plt.rcParams.update(RCPARAMS)
fig1, ax1 = plt.subplots(figsize=(9, 4))
bars = ax1.bar(
    counts.index, counts.values,
    color=[COLORS[l] for l in counts.index],
    width=0.6, edgecolor="white",
)
for bar in bars:
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        str(int(bar.get_height())),
        ha="center", va="bottom", fontsize=10,
    )
ax1.axhline(
    counts.mean(), color="gray", linestyle="--", linewidth=1.2,
    label=f"Rata-rata = {counts.mean():.0f}",
)
ax1.set_xlabel("Label Emosi")
ax1.set_ylabel("Jumlah Data")
ax1.set_title("Jumlah Data per Label Emosi", fontweight="bold")
ax1.legend(fontsize=9)
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

dist_df = counts.to_frame("jumlah").assign(
    persen=lambda x: (x["jumlah"] / len(df_filtered) * 100).round(1)
)
dist_df.index.name = "label"
dist_df.columns    = ["Jumlah", "Persen (%)"]

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(dist_df, use_container_width=True)
with col_b:
    selisih = int(counts.max() - counts.min())
    if selisih <= 5:
        st.success(f"✅ Dataset **seimbang** (selisih max-min = {selisih} data). Tidak perlu oversampling.")
    elif selisih <= 50:
        st.warning(f"⚠️ Dataset **agak tidak merata** (selisih = {selisih}). Pertimbangkan balancing ringan.")
    else:
        st.error(f"❌ Dataset **tidak seimbang** (selisih = {selisih}). Perlu oversampling/undersampling.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Q2 — Panjang teks
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Q2 — Apakah panjang teks berbeda-beda antar emosi?")
st.markdown(
    "**Kenapa penting:** Variasi panjang teks antar label bisa jadi sinyal (fitur tambahan) "
    "untuk model, atau bisa menjadi noise jika tidak ditangani."
)

wc_mean = df_filtered.groupby("label")["word_count"].mean().reindex(order_filtered).dropna()

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))

bars2 = axes2[0].barh(
    wc_mean.index, wc_mean.values,
    color=[COLORS[l] for l in wc_mean.index],
    edgecolor="white",
)
for bar, val in zip(bars2, wc_mean.values):
    axes2[0].text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}", va="center", fontsize=9,
    )
axes2[0].set_title("Rata-rata Jumlah Kata per Label", fontweight="bold")
axes2[0].set_xlabel("Rata-rata Word Count")
axes2[0].invert_yaxis()

data_groups = [df_filtered[df_filtered["label"] == l]["word_count"].values for l in order_filtered]
bp = axes2[1].boxplot(
    data_groups, labels=order_filtered, patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
)
for patch, label in zip(bp["boxes"], order_filtered):
    patch.set_facecolor(COLORS[label])
    patch.set_alpha(0.75)
axes2[1].set_title("Distribusi Word Count per Label", fontweight="bold")
axes2[1].set_xlabel("Label")
axes2[1].set_ylabel("Jumlah Kata")
axes2[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

stats = (
    df_filtered.groupby("label")["word_count"]
    .agg(["mean", "min", "max"])
    .round(1)
    .reindex(order_filtered)
    .dropna()
)
stats.columns    = ["Mean", "Min", "Max"]
stats.index.name = "label"

col_c, col_d = st.columns([1, 2])
with col_c:
    st.dataframe(stats, use_container_width=True)
with col_d:
    longest  = stats["Mean"].idxmax()
    shortest = stats["Mean"].idxmin()
    st.info(
        f"📏 **{longest.upper()}** rata-rata teks terpanjang ({stats.loc[longest,'Mean']:.1f} kata)  \n"
        f"📏 **{shortest.upper()}** rata-rata teks terpendek ({stats.loc[shortest,'Mean']:.1f} kata)  \n\n"
        "→ `word_count` bisa dijadikan fitur numerik tambahan saat training."
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Q3 — Kata dominan
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Q3 — Kata apa yang paling sering muncul per emosi?")
st.markdown(
    "**Kenapa penting:** Kata-kata khas per emosi adalah sinyal utama untuk model. "
    "Kata yang tumpang tindih antar emosi menandakan model perlu representasi lebih kaya."
)

n_cols = min(4, len(order_filtered))
n_rows = -(-len(order_filtered) // n_cols)

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4))
axes3 = [axes3] if len(order_filtered) == 1 else axes3.flatten()

for i, label in enumerate(order_filtered):
    texts = df_filtered[df_filtered["label"] == label]["text"].tolist()
    kws   = top_words(texts, n=8)
    words = [k[0] for k in kws[::-1]]
    freqs = [k[1] for k in kws[::-1]]

    axes3[i].barh(words, freqs, color=COLORS[label], alpha=0.82, edgecolor="white")
    axes3[i].set_title(label.capitalize(), fontweight="bold",
                       color=COLORS[label], fontsize=11)
    axes3[i].set_xlabel("Frekuensi")
    axes3[i].tick_params(axis="y", labelsize=9)

for j in range(len(order_filtered), len(axes3)):
    axes3[j].set_visible(False)

plt.suptitle("Top 8 Kata per Label Emosi (stopword dihapus)",
             fontweight="bold", fontsize=13)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

st.info(
    "→ Kata yang **unik** di satu emosi = sinyal kuat untuk model.  \n"
    "→ Kata yang muncul di **banyak emosi** sebaiknya diberi bobot lebih rendah dengan TF-IDF."
)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Kesimpulan
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📝 Kesimpulan")
st.table(
    pd.DataFrame({
        "Pertanyaan": [
            "Q1 — Distribusi kelas",
            "Q2 — Panjang teks",
            "Q3 — Kata dominan",
        ],
        "Temuan": [
            "Hampir seimbang (198–200 per kelas)",
            "Berbeda signifikan antar emosi",
            "Tiap emosi punya kata khasnya",
        ],
        "Implikasi": [
            "Tidak perlu oversampling, langsung bisa training",
            "word_count bisa dijadikan fitur tambahan",
            "TF-IDF atau BoW cukup sebagai baseline fitur",
        ],
    })
)

st.caption("EDA — Dataset Emosi Markus Erelius | dibuat dengan Streamlit")