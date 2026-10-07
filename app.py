import streamlit as st
import joblib
import numpy as np
import string
import re
import requests
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# LOAD MODEL DAN SASTRAWI (Gunakan Cache)
@st.cache_resource
def load_models_and_tools():
    # Load Model Machine Learning
    skipgram_model = Word2Vec.load("model_streamlit/best_skipgram.model")
    nb_classifier = joblib.load("model_streamlit/best_naive_bayes.joblib")
    
    # Load Sastrawi Tools
    stopword_factory = StopWordRemoverFactory()
    stopwords = set(stopword_factory.get_stop_words())
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    return skipgram_model, nb_classifier, stopwords, stemmer

skipgram_model, nb_classifier, stopwords, stemmer = load_models_and_tools()


# FUNGSI WEB SCRAPING BERITA DETIK
def scrape_detik(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Berita detik biasanya menyimpan teks utama di class "detail__body-text"
        article = soup.find('div', class_='detail__body-text')
        
        if article:
            # Mengambil semua teks paragraf
            paragraphs = article.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            return text
        else:
            return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat scraping: {e}")
        return None

# FUNGSI PREPROCESSING & VEKTORISASI
def preprocess_text(text, stopwords, stemmer):
    text = str(text).lower()
    
    # Cleaning
    text = re.sub(r'\([a-z]{2,5}/[a-z]{2,5}\)', ' ', text, flags=re.IGNORECASE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenisasi
    tokens = text.split()
    
    # Stopword Removal
    tokens = [word for word in tokens if word not in stopwords]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

def get_document_vector(model, tokens):
    # Rata-rata vektor kata dalam satu dokumen
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# ANTARMUKA (UI) STREAMLIT
st.set_page_config(page_title="Klasifikasi Berita Detik", page_icon="📰")
st.title("📰 Klasifikasi Berita Detik")
st.write("Sistem ini akan mengambil isi berita dari link Detik (Finance/Sport) dan mengklasifikasikannya menggunakan model Skip-gram & Naive Bayes.")

# Input Link
url_input = st.text_input("🔗 Masukkan Link Berita Detik:")

if st.button("Proses Berita"):
    if url_input.strip():
        with st.spinner("Sedang mengambil isi berita..."):
            isi_berita = scrape_detik(url_input)
            
        if isi_berita:
            with st.spinner("Sedang memproses teks dan memprediksi..."):
                # 1. Preprocessing teks
                tokens = preprocess_text(isi_berita, stopwords, stemmer)
                teks_preprocessed = " ".join(tokens) # Menggabungkan token menjadi kalimat utuh
                
                # 2. Vektorisasi dokumen
                doc_vector = get_document_vector(skipgram_model, tokens)
                
                # 3. Prediksi Label
                prediction = nb_classifier.predict([doc_vector])
                
                # 4. Prediksi Probabilitas (Persentase)
                probabilities = nb_classifier.predict_proba([doc_vector])[0]
                classes = nb_classifier.classes_
                
            # MENAMPILKAN HASIL KE LAYAR
            st.success(f"### 🏆 Hasil Klasifikasi: **{prediction[0].upper()}**")
            
            st.markdown("---")
            st.markdown("### 📊 Persentase (Probabilitas) Label")
            # Menampilkan persentase untuk setiap class (misal: Finance dan Sport)
            col1, col2 = st.columns(2)
            for i, (label, prob) in enumerate(zip(classes, probabilities)):
                persentase = prob * 100
                if i % 2 == 0:
                    col1.metric(label=label.upper(), value=f"{persentase:.2f}%")
                else:
                    col2.metric(label=label.upper(), value=f"{persentase:.2f}%")

            st.markdown("---")
            st.markdown("### 📝 Detail Teks")
            
            # Gunakan Expander agar UI tetap rapi
            with st.expander("📄 Lihat Isi Berita Asli", expanded=True):
                st.write(isi_berita)
                
            with st.expander("⚙️ Lihat Hasil Preprocessing", expanded=True):
                st.write(teks_preprocessed)
                
        else:
            st.warning("Gagal mengambil teks berita. Pastikan link yang dimasukkan adalah link artikel berita Detik yang valid.")
    else:
        st.error("Silakan masukkan link terlebih dahulu!")