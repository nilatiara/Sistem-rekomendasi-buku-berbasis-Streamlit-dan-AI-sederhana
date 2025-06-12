import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
books = pd.read_csv("books.csv")
transactions = pd.read_csv("transactions.csv")

# Gabung transaksi dengan buku
data = transactions.merge(books, on="book_id")

# TF-IDF untuk kategori
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['category'])
cos_sim = cosine_similarity(tfidf_matrix)

# Fungsi rekomendasi
def get_recommendations(user_id, top_n=5):
    user_data = data[data['user_id'] == user_id]
    user_books = user_data['book_id'].tolist()

    if not user_books:
        return pd.DataFrame(columns=['title', 'category'])

    # Skor rata-rata similarity
    scores = cos_sim[user_books].mean(axis=0)
    scores_indices = scores.argsort()[::-1]

    recommended_books = books.iloc[scores_indices]
    recommended_books = recommended_books[~recommended_books['book_id'].isin(user_books)]

    return recommended_books.head(top_n)[['title', 'category']]

# UI
st.title("📚 SmartLibRec – Rekomendasi Buku Perpustakaan")
user_id = st.number_input("Masukkan ID Pengguna:", min_value=1, step=1)

if st.button("Tampilkan Rekomendasi"):
    recs = get_recommendations(user_id)
    if recs.empty:
        st.warning("Belum ada riwayat peminjaman.")
    else:
        st.success("Berikut adalah buku yang direkomendasikan:")
        st.table(recs)
