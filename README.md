# EDA Emosi Markus Erelius — Streamlit App

Dashboard EDA interaktif untuk dataset klasifikasi emosi berbahasa Indonesia.

## Cara Pakai

**1. Install dependensi**
```bash
pip install -r requirements.txt
```

**2. Jalankan aplikasi**
```bash
streamlit run app.py
```

**3. Upload dataset**  
Klik tombol upload di halaman, pilih file `dataset_markus_erelius_clean.csv`.

---

## Pertanyaan Bisnis yang Dijawab

| # | Pertanyaan | Visualisasi |
|---|---|---|
| Q1 | Apakah jumlah data tiap emosi merata? | Bar chart distribusi |
| Q2 | Apakah panjang teks berbeda antar emosi? | Bar rata-rata + Boxplot |
| Q3 | Kata apa yang paling sering muncul per emosi? | Horizontal bar per label |

## Fitur Interaktif

- **Filter label** — pilih emosi mana yang ingin dianalisis via sidebar
- **Metrik otomatis** — status keseimbangan dataset terdeteksi otomatis
- **Tabel statistik** — ringkasan word count per label

## Requirement Dataset

File CSV harus memiliki dua kolom:
- `label` — nama emosi (netral, senang, sedih, marah, takut, jijik, kaget, cape)
- `text`  — teks kutipan
