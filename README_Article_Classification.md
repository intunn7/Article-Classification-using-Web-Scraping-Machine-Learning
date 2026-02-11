# Article Classification using Web Scraping & Machine Learning 📊🤖

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-completed-success.svg)]()

Sistem klasifikasi otomatis artikel ilmiah menggunakan web scraping dari Springer dan machine learning untuk mengkategorikan artikel ke dalam 5 topik: Audio Processing, Video Processing, Signal Processing, Image Processing, dan Text Processing.

![Project Banner](https://via.placeholder.com/800x200/4285f4/ffffff?text=Article+Classification+ML+Project)

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

## 📁 Struktur Project

```
article-classification/
├── fixx.ipynb                      # Main Jupyter notebook
├── scraping_totalpages.csv         # Raw scraped data
├── hasil_preprocessing.csv         # Preprocessed data
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation (file ini)
├── models/                         # Saved models (optional)
│   ├── best_model_tfidf_lr.pkl
│   ├── vectorizer_tfidf.pkl
│   └── label_encoder.pkl
├── visualizations/                 # Generated plots (optional)
│   ├── confusion_matrix.png
│   ├── wordcloud_*.png
│   └── distribution_*.png
└── data/                          # Additional data (optional)
    └── raw_articles.json
```

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

## 🐛 Troubleshooting

### Error: "HTTP 429 - Too Many Requests"
**Penyebab**: Springer membatasi request rate

**Solusi**:
```python
import time
time.sleep(2)  # Tambahkan delay antar request
```

### Error: "NLTK data not found"
**Solusi**:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Error: "Memory Error during vectorization"
**Penyebab**: Dataset terlalu besar untuk RAM

**Solusi**:
- Kurangi jumlah artikel
- Gunakan `max_features` parameter:
```python
vectorizer = TfidfVectorizer(max_features=5000)
```

### Warning: "Convergence Warning - LogisticRegression"
**Solusi**:
```python
LogisticRegression(max_iter=200)  # Increase iterations
```

### Scraping Returns Empty Data
**Penyebab**: Struktur HTML Springer berubah

**Solusi**:
- Inspect halaman web terbaru
- Update CSS selectors dalam kode scraping
- Gunakan browser developer tools untuk debugging

---

## 🚧 Roadmap & Future Improvements

### Version 2.0 (Planned)
- [ ] Deep Learning models (BERT, Transformer)
- [ ] Real-time classification API
- [ ] Support multi-language articles
- [ ] Automatic model retraining pipeline
- [ ] Web dashboard untuk visualisasi
- [ ] Integration dengan database (PostgreSQL/MongoDB)
- [ ] Citation network analysis
- [ ] Author collaboration network

### Enhancement Ideas
- [ ] Active learning untuk improve model
- [ ] Ensemble methods
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning (GridSearch/RandomSearch)
- [ ] Cross-validation dengan K-Fold
- [ ] Export model ke ONNX untuk deployment

---

## 📊 Evaluation Metrics Explained

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Persentase prediksi yang benar dari total prediksi.

### Precision
```
Precision = TP / (TP + FP)
```
Dari yang diprediksi positif, berapa yang benar-benar positif?

### Recall
```
Recall = TP / (TP + FN)
```
Dari yang sebenarnya positif, berapa yang berhasil terdeteksi?

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean dari precision dan recall.

### Confusion Matrix
Tabel yang menunjukkan:
- **True Positive (TP)**: Prediksi benar positif
- **True Negative (TN)**: Prediksi benar negatif
- **False Positive (FP)**: Prediksi salah positif
- **False Negative (FN)**: Prediksi salah negatif

---

## 👥 Tim Pengembang

Project ini dikembangkan oleh:

- **[Nama Anggota 1]** - Data Scientist & ML Engineer
- **[Nama Anggota 2]** - Web Scraping Specialist
- **[Nama Anggota 3]** - Data Analyst & Visualization
- **[Nama Anggota 4]** - Documentation & Testing

*Silakan update dengan informasi tim Anda*

---

## 🤝 Kontribusi

Kontribusi sangat dihargai! Cara berkontribusi:

1. **Fork** repository ini
2. Buat **branch** fitur (`git checkout -b feature/AmazingFeature`)
3. **Commit** perubahan (`git commit -m 'Add some AmazingFeature'`)
4. **Push** ke branch (`git push origin feature/AmazingFeature`)
5. Buat **Pull Request**

### Contribution Guidelines:
- ✅ Gunakan PEP 8 style guide
- ✅ Tambahkan docstrings untuk fungsi baru
- ✅ Update README jika menambah fitur
- ✅ Test kode sebelum submit PR
- ✅ Sertakan komentar yang jelas

---

## 📄 Lisensi

Project ini dilisensikan under **MIT License**.

```
MIT License

Copyright (c) 2024 Article Classification Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 📧 Kontak & Support

Punya pertanyaan atau saran?

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/article-classification/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/article-classification/discussions)

---

## 🌟 Acknowledgments

Terima kasih kepada:
- **Springer Nature** untuk menyediakan akses artikel ilmiah
- **Scikit-learn Team** untuk machine learning library yang powerful
- **NLTK Team** untuk NLP tools
- **Python Community** untuk semua library yang luar biasa

---

## 📚 References & Resources

### Papers & Articles:
- [Text Classification with Machine Learning](https://example.com)
- [TF-IDF Feature Extraction](https://example.com)
- [Naive Bayes for Text Classification](https://example.com)

### Documentation:
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK Documentation](https://www.nltk.org/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Tutorials:
- [Text Mining with Python](https://example.com)
- [Web Scraping Best Practices](https://example.com)

---

## 📝 Citation

Jika Anda menggunakan project ini dalam penelitian, mohon cite:

```bibtex
@misc{article_classification_2024,
  author = {Your Name},
  title = {Article Classification using Web Scraping and Machine Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/article-classification}
}
```

---

## ⚠️ Disclaimer

- Project ini dibuat untuk tujuan **edukasi dan penelitian**
- Web scraping harus mematuhi **Terms of Service** Springer
- Gunakan data scraping secara **etis dan bertanggung jawab**
- Tidak untuk keperluan komersial tanpa izin

---

## 📊 Project Statistics

- **Total Lines of Code**: ~500 lines
- **Total Cells**: 20+ cells
- **Dataset Size**: ~744 articles
- **Feature Dimensions**: ~10,000+ features
- **Models Trained**: 9 combinations
- **Development Time**: [Duration]

---

<div align="center">

### ⭐ Jika project ini bermanfaat, berikan **Star**! ⭐

**Made with ❤️ and ☕ by the Article Classification Team**

[⬆ Back to Top](#article-classification-using-web-scraping--machine-learning-)

</div>
