# Article Classification using Web Scraping & Machine Learning 📊🤖

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Sistem klasifikasi otomatis artikel ilmiah menggunakan web scraping dari Springer dan machine learning untuk mengkategorikan artikel ke dalam 5 topik: Audio Processing, Video Processing, Signal Processing, Image Processing, dan Text Processing.
---

## 📋 Deskripsi Project

Project ini merupakan implementasi **end-to-end machine learning pipeline** yang mencakup:
1. **Web Scraping**: Mengumpulkan data artikel ilmiah dari Springer (2019-2024)
2. **Data Preprocessing**: Cleaning, tokenization, lemmatization
3. **Feature Engineering**: One-Hot Encoding, Bag-of-Words, TF-IDF
4. **Model Training**: Naive Bayes, SVM, Logistic Regression
5. **Evaluation**: Perbandingan performa model dengan berbagai metrik
6. **Visualization**: Analisis distribusi data dan hasil klasifikasi

### 🎯 Tujuan Project
- Mengotomatisasi klasifikasi artikel ilmiah berdasarkan topik
- Membandingkan efektivitas berbagai teknik feature extraction
- Mengevaluasi performa algoritma machine learning untuk text classification
- Menganalisis tren publikasi artikel dalam 5 tahun terakhir (2019-2024)

---

## ✨ Fitur Utama

### 🕷️ Web Scraping
- **Sumber Data**: Springer Link (link.springer.com)
- **Periode**: 2019 - 2024
- **Sampling**: 0.1% dari total artikel per topik (proporsional)
- **Data Dikumpulkan**: 
  - Judul artikel
  - Tahun publikasi
  - Abstrak
  - Topik/kategori

### 🔧 Data Preprocessing
- **Text Cleaning**: Remove punctuation, lowercase conversion
- **Tokenization**: Split text into words
- **Stopwords Removal**: Hapus kata-kata umum (English)
- **Lemmatization**: Normalisasi kata ke bentuk dasar

### 📊 Feature Engineering
Tiga metode ekstraksi fitur:
1. **One-Hot Encoding**: Binary presence/absence
2. **Bag-of-Words (BoW)**: Word frequency counting
3. **TF-IDF**: Term Frequency-Inverse Document Frequency

### 🤖 Machine Learning Models
- **Naive Bayes** (MultinomialNB)
- **Support Vector Machine** (SVM - Linear Kernel)
- **Logistic Regression** (C=10, max_iter=100)

### 📈 Visualisasi & Analisis
- Distribusi artikel per topik
- Tren publikasi per tahun
- Word Cloud untuk setiap topik
- Confusion Matrix untuk evaluasi model
- Perbandingan akurasi model

---

## 🛠️ Teknologi yang Digunakan

| Kategori | Library/Tool | Fungsi |
|----------|--------------|--------|
| **Core** | Python 3.9+ | Bahasa pemrograman |
| **Notebook** | Jupyter Notebook | Development environment |
| **Web Scraping** | BeautifulSoup | HTML parsing |
| | requests | HTTP requests |
| **Data Processing** | pandas | Data manipulation |
| | numpy | Numerical operations |
| **NLP** | NLTK | Natural language processing |
| | WordNetLemmatizer | Word lemmatization |
| **Feature Extraction** | CountVectorizer | BoW & One-Hot |
| | TfidfVectorizer | TF-IDF calculation |
| **Machine Learning** | scikit-learn | ML algorithms & metrics |
| | LogisticRegression | Classification model |
| | SVC (SVM) | Classification model |
| | MultinomialNB | Classification model |
| **Visualization** | matplotlib | Data visualization |
| | seaborn | Statistical plots |
| | WordCloud | Word cloud generation |
| **Model Persistence** | joblib | Save/load models |

---

## 📦 Instalasi

### Persyaratan Sistem
- Python 3.9 atau lebih baru
- Jupyter Notebook / JupyterLab
- Koneksi internet (untuk web scraping)
- RAM minimum 4GB (recommended 8GB)

### Langkah Instalasi

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/article-classification.git
cd article-classification
```

#### 2️⃣ Buat Virtual Environment (Opsional tapi Disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Atau install manual:**
```bash
pip install pandas numpy requests beautifulsoup4 nltk matplotlib seaborn scikit-learn wordcloud joblib jupyter
```

#### 4️⃣ Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

#### 5️⃣ Jalankan Jupyter Notebook
```bash
jupyter notebook fixx.ipynb
```

---

## 🚀 Cara Menggunakan

### Pipeline Lengkap:

#### **Step 1: Web Scraping**
```python
# Jalankan cell scraping untuk mengumpulkan data
# Output: scraping_totalpages.csv
```
- Scrapes artikel dari 5 topik
- Sampling proporsional 0.1% per topik
- Menyimpan hasil ke CSV

#### **Step 2: Data Preprocessing**
```python
# Jalankan cell preprocessing
# Output: hasil_preprocessing.csv
```
- Cleaning text
- Tokenization & lemmatization
- Remove stopwords

#### **Step 3: Feature Engineering**
```python
# Jalankan cell feature engineering
```
- Membuat 3 representasi fitur:
  - One-Hot Encoding
  - Bag-of-Words
  - TF-IDF

#### **Step 4: Model Training & Evaluation**
```python
# Jalankan cell training
```
- Split data (80% train, 20% test)
- Train 3 model untuk setiap metode feature
- Total: 9 kombinasi model-feature

#### **Step 5: Analisis & Visualisasi**
```python
# Jalankan cell visualisasi
```
- Confusion matrix
- Akurasi comparison
- Word clouds
- Distribusi topik per tahun

---

## 📊 Dataset

### Topik yang Dikumpulkan:
1. **Audio Processing** (~22 artikel)
2. **Video Processing** (~61 artikel)
3. **Signal Processing** (~209 artikel)
4. **Image Processing** (~231 artikel)
5. **Text Processing** (~221 artikel)

### Struktur Data:
| Kolom | Deskripsi | Tipe |
|-------|-----------|------|
| Topik | Kategori artikel | String |
| Judul | Judul artikel | String |
| Tahun Terbit | Tahun publikasi (2019-2024) | Integer |
| Abstrak | Abstrak artikel | Text |
| Preprocessing Judul | Judul yang sudah di-preprocess | String |
| Preprocessing Abstrak | Abstrak yang sudah di-preprocess | Text |

### Sample Data:
```
Topik: Audio Processing
Judul: "A lightweight approach to real-time speaker diarization"
Tahun: 2024
Abstrak: "This manuscript deals with the task of real-time speaker diarization..."
```

---

## 🎯 Hasil & Performa Model

### Model Comparison (Berdasarkan Akurasi)

| Feature Method | Naive Bayes | SVM | Logistic Regression |
|----------------|-------------|-----|---------------------|
| **One-Hot Encoding** | ~XX% | ~XX% | ~XX% |
| **Bag-of-Words** | ~XX% | ~XX% | ~XX% |
| **TF-IDF** | ~XX% | ~XX% | **~XX%** ⭐ |

*Note: Jalankan notebook untuk melihat hasil aktual*

### Key Findings:
- 🏆 **Best Model**: [Model terbaik berdasarkan hasil]
- 📈 **Best Feature Method**: TF-IDF umumnya memberikan hasil terbaik
- 🎯 **Average Accuracy**: ~XX%
- 📊 **Class Performance**: Image/Signal Processing cenderung lebih mudah diklasifikasi

---

## 📈 Visualisasi

### 1. Distribusi Artikel per Topik
Bar chart menampilkan jumlah artikel yang berhasil di-scrape untuk setiap topik.

### 2. Tren Publikasi per Tahun
Line plot menunjukkan perkembangan jumlah publikasi dari 2019-2024 untuk setiap topik.

### 3. Word Cloud per Topik
Visual representasi kata-kata paling sering muncul dalam setiap kategori topik.

### 4. Confusion Matrix
Heatmap yang menunjukkan performa prediksi model untuk setiap kelas.

### 5. Model Accuracy Comparison
Bar chart perbandingan akurasi antar model dan metode feature extraction.

---

## 🔧 Konfigurasi & Parameter

### Web Scraping Parameters:
```python
SAMPLE_PERCENTAGE = 0.001  # 0.1% dari total artikel
ARTICLES_PER_PAGE = 20
DATE_RANGE = "2019-2024"
LANGUAGE = "En"
```

### Model Hyperparameters:

**Logistic Regression:**
```python
C = 10
max_iter = 100
random_state = 42
```

**SVM:**
```python
kernel = 'linear'
random_state = 42
```

**Naive Bayes:**
```python
# Default parameters
```

### Train-Test Split:
```python
test_size = 0.2
random_state = 42
```

---

---

## 🧪 Metodologi

### 1. Data Collection
- **Source**: Springer Link API/Web interface
- **Method**: BeautifulSoup HTML parsing
- **Sampling**: Stratified proportional sampling (0.1%)

### 2. Preprocessing Pipeline
```
Raw Text → Lowercase → Remove Punctuation → Tokenize 
→ Remove Stopwords → Lemmatization → Clean Text
```

### 3. Feature Extraction
- **One-Hot**: Binary encoding kata unik
- **BoW**: Frequency count per kata
- **TF-IDF**: Weighted importance per kata

### 4. Model Training
- **Cross-validation**: 80-20 split
- **Evaluation Metrics**: 
  - Accuracy
  - Precision (per class)
  - Recall (per class)
  - F1-Score
  - Confusion Matrix

### 5. Model Selection
- Perbandingan 9 kombinasi (3 models × 3 features)
- Pilih model dengan akurasi tertinggi
- Save model terbaik dengan joblib

---

## 💡 Use Cases

### 1. 📚 Digital Library Management
- Auto-tagging artikel ilmiah
- Kategorisasi otomatis paper baru
- Rekomendasi artikel serupa

### 2. 🔍 Research Assistant
- Filtering artikel berdasarkan topik
- Literature review automation
- Trend analysis penelitian

### 3. 📊 Academic Analytics
- Analisis publikasi per bidang
- Identifikasi hot topics
- Research gap detection

### 4. 🎓 Educational Platform
- Kurasi konten pembelajaran
- Resource recommendation
- Topic clustering

---
