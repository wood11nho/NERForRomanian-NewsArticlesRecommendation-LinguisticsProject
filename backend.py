from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Load spaCy model for Romanian lemmatization and stopword removal
nlp = spacy.load("ro_core_news_sm")

# Load SentenceTransformer model for embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load dataset
news_data = pd.read_csv("NewsArticles/entities_and_relations.csv")
news_data.fillna("", inplace=True)

# Preprocess text: Lemmatization and Stopword Removal
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Apply preprocessing to cleaned_content
news_data["preprocessed_content"] = news_data["cleaned_content"].apply(preprocess_text)
news_data["preprocessed_title"] = news_data["Title"].apply(preprocess_text)
news_data["preprocessed_summary"] = news_data["Summary"].apply(preprocess_text)

# Compute embeddings for preprocessed_content, title, and summary
content_embeddings = model.encode(news_data["preprocessed_content"].tolist())
title_embeddings = model.encode(news_data["preprocessed_title"].tolist())
summary_embeddings = model.encode(news_data["preprocessed_summary"].tolist())

# Fetch unique categories dynamically
unique_categories = news_data["Category"].unique().tolist()

# Model for search queries
class SearchQuery(BaseModel):
    query: str
    filters: List[str] = []
    
# Helper to compute TF-IDF similarity
def compute_tfidf_similarity(query, corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return similarity_scores

# Helper to compute embedding similarity
def compute_embedding_similarity(query_embedding, content_embeddings):
    similarity_scores = cosine_similarity([query_embedding], content_embeddings).flatten()
    return similarity_scores

@app.get("/categories")
def get_categories():
    try:
        logger.info("Fetching categories...")
        return {"categories": unique_categories}
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return {"error": "Unable to fetch categories."}

@app.post("/search")
def search_news(search_query: SearchQuery):
    query = search_query.query
    filters = search_query.filters

    # Preprocess the query
    preprocessed_query = preprocess_text(query)
    query_embedding = model.encode([preprocessed_query])[0]

    filtered_data = news_data.copy()

    # Check if the query matches one of the categories
    if query.lower() in [category.lower() for category in unique_categories]:
        filtered_data = filtered_data[filtered_data["Category"].str.contains(query, case=False)]
        filtered_data["similarity_percent"] = 100.0  # Set similarity to 100% for category match
    else:
        # Apply filters if categories are selected
        if filters:
            for f in filters:
                filtered_data = filtered_data[filtered_data["Category"].str.contains(f, case=False)]

        # Get indices of the filtered data to align embeddings
        filtered_indices = filtered_data.index.tolist()
        filtered_content_embeddings = content_embeddings[filtered_indices]
        filtered_title_embeddings = title_embeddings[filtered_indices]
        filtered_summary_embeddings = summary_embeddings[filtered_indices]

        # Compute TF-IDF similarity with cleaned_content
        corpus = filtered_data["preprocessed_content"].tolist()
        tfidf_similarity_scores = compute_tfidf_similarity(preprocessed_query, corpus)

        # Compute embedding similarity with cleaned_content, title, and summary
        content_similarity_scores = compute_embedding_similarity(query_embedding, filtered_content_embeddings)
        title_similarity_scores = compute_embedding_similarity(query_embedding, filtered_title_embeddings)
        summary_similarity_scores = compute_embedding_similarity(query_embedding, filtered_summary_embeddings)

        # Combine similarities
        combined_similarity = (
            0.4 * tfidf_similarity_scores +          # 40% weight for TF-IDF similarity
            0.3 * content_similarity_scores +        # 30% weight for cleaned_content embeddings
            0.2 * title_similarity_scores +          # 20% weight for title embeddings
            0.1 * summary_similarity_scores          # 10% weight for summary embeddings
        )

        # Boost similarity if the selected category matches the article's category
        boost_factor = 1.2  # Boost similarity by 20% for category match
        filtered_data["similarity"] = combined_similarity
        filtered_data["similarity_percent"] = filtered_data.apply(
            lambda row: min(100.0, row["similarity"] * 100 * boost_factor)
            if filters and row["Category"] in filters
            else row["similarity"] * 100,
            axis=1,
        )

    # Add metrics to the response
    results = []
    for idx, row in filtered_data.iterrows():
        results.append({
            "Title": row["Title"],
            "Category": row["Category"],
            "Content": row["Content"],
            "similarity_percent": float(row["similarity_percent"]),
            "tfidf_similarity": float(tfidf_similarity_scores[idx]),
            "content_similarity": float(content_similarity_scores[idx]),
            "title_similarity": float(title_similarity_scores[idx]),
            "summary_similarity": float(summary_similarity_scores[idx]),
            "combined_similarity": float(combined_similarity[idx]),
            "boosted_similarity": float(row["similarity_percent"]) / 100.0
        })
        
    # Return top 10 results
    results = sorted(results, key=lambda x: x["similarity_percent"], reverse=True)[:10]
    return {"results": results}