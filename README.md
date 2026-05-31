# Analisis Emosi pada Kutipan Marcus Aurelius

## Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis emosi yang terkandung dalam kumpulan kutipan Marcus Aurelius menggunakan pendekatan Exploratory Data Analysis (EDA) dan visualisasi data interaktif dengan Streamlit.

Analisis dilakukan untuk memahami distribusi emosi, karakteristik panjang teks, serta kata-kata yang dominan pada setiap kategori emosi.

---

## Business Understanding

### Latar Belakang

Marcus Aurelius dikenal sebagai filsuf Stoik yang meninggalkan banyak kutipan mengenai kehidupan, pengendalian diri, kebijaksanaan, dan ketenangan batin. Dengan melakukan analisis terhadap kumpulan kutipan tersebut, kita dapat memperoleh pemahaman mengenai pola emosi yang muncul dalam teks.

### Pertanyaan Bisnis

1. Apakah jumlah data pada setiap kategori emosi sudah merata?
2. Apakah panjang teks berbeda pada setiap kategori emosi?
3. Kata apa yang paling sering muncul pada setiap kategori emosi?

---

## Dataset

Dataset terdiri dari dua kolom utama:

| Kolom | Deskripsi                   |
| ----- | --------------------------- |
| label | Kategori emosi              |
| text  | Isi kutipan Marcus Aurelius |

### Data Dictionary

| Kolom       | Tipe Data | Deskripsi                    |
| ----------- | --------- | ---------------------------- |
| label       | object    | Label emosi                  |
| text        | object    | Isi kutipan                  |
| word_count  | integer   | Jumlah kata pada kutipan     |
| char_length | integer   | Jumlah karakter pada kutipan |

---

## Tahapan Analisis

### 1. Data Understanding

* Memuat dataset
* Memeriksa struktur data
* Memeriksa missing values
* Membuat data dictionary

### 2. Data Preparation

* Membersihkan nama kolom
* Menghapus data kosong
* Membuat fitur:

  * word_count
  * char_length

### 3. Exploratory Data Analysis (EDA)

Analisis dilakukan untuk menjawab seluruh pertanyaan bisnis.

#### Distribusi Emosi

Menampilkan jumlah data pada setiap kategori emosi.

#### Analisis Panjang Teks

Menggunakan:

* Word Count
* Character Length

untuk membandingkan panjang teks antar emosi.

#### Analisis Kata Dominan

Menggunakan:

* Top 10 Kata Terbanyak
* Word Cloud

untuk mengetahui kata yang sering muncul pada setiap emosi.

---

## Dashboard Streamlit

Dashboard interaktif dikembangkan menggunakan Streamlit dengan beberapa halaman:

### Overview

* Jumlah data
* Jumlah kategori emosi
* Preview dataset
* Data dictionary

### Distribusi Emosi

Menampilkan persebaran jumlah data pada setiap kategori emosi.

### Analisis Panjang Teks

Menampilkan boxplot:

* Word Count
* Character Length

### Kata Dominan

Menampilkan:

* Word Cloud
* Top Kata Terbanyak

### Kesimpulan

Menampilkan insight utama hasil analisis.

---

## Tools dan Library

* Python
* Pandas
* Matplotlib
* Seaborn
* WordCloud
* Streamlit

---

## Instalasi

Clone repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Menjalankan Dashboard

```bash
streamlit run dashboard.py
```

---

## Struktur Proyek

```text
project/
│
├── dashboard.py
├── requirements.txt
├── dataset_markus_erelius_clean.csv
├── EDA_Emosi_Marcus_Aurelius.ipynb
├── README.md
│
└── assets/
```

---

## Hasil Analisis

Berdasarkan analisis yang dilakukan:

* Distribusi emosi dapat digunakan untuk melihat keseimbangan dataset.
* Panjang teks berbeda pada beberapa kategori emosi.
* Kata-kata tertentu muncul lebih dominan pada emosi tertentu.
* Dashboard Streamlit mempermudah eksplorasi dan interpretasi data secara interaktif.

---

## Author

Capstone Project – Data Analyst

Analisis Emosi pada Kutipan Marcus Aurelius menggunakan Python, EDA, dan Streamlit.
