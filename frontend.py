import streamlit as st
import requests
import matplotlib.pyplot as plt

# App title
st.title("Romanian News App with NER")

# Fetch available categories from the backend
categories_response = requests.get("http://localhost:8000/categories")
categories = categories_response.json()["categories"]

# Search bar
query = st.text_input("Search for news articles")

# Filters
selected_categories = st.multiselect(
    "Filter by category", 
    options=categories
)

# State to store search results and visibility toggles
if "results" not in st.session_state:
    st.session_state.results = []
if "show_metrics" not in st.session_state:
    st.session_state.show_metrics = {}

# Search button
if st.button("Search"):
    # Send request to FastAPI backend
    response = requests.post(
        "http://localhost:8000/search",
        json={"query": query, "filters": selected_categories},
    )
    results = response.json()["results"]

    # Store results and initialize visibility toggles
    st.session_state.results = results
    st.session_state.show_metrics = {idx: False for idx in range(len(results))}

# Display results
st.subheader("Top 10 Relevant News")
for idx, result in enumerate(st.session_state.results):
    st.write(f"**{result['Title']}**")
    st.write(f"*Category: {result['Category']}*")
    st.write(f"**Relevance:** {result['similarity_percent']}%")
    st.write(result["Content"])

    # Toggle button for metrics visibility
    if st.button(f"Show/Hide Metrics for Article {idx}"):
        st.session_state.show_metrics[idx] = not st.session_state.show_metrics[idx]

    # Display metrics if toggled on
    if st.session_state.show_metrics[idx]:
        metrics = {
            "TF-IDF Similarity": result.get("tfidf_similarity", 0),
            "Content Embedding Similarity": result.get("content_similarity", 0),
            "Title Embedding Similarity": result.get("title_similarity", 0),
            "Summary Embedding Similarity": result.get("summary_similarity", 0),
            "Combined Similarity": result.get("combined_similarity", 0),
            "Boosted Similarity": result.get("boosted_similarity", 0),
        }

        labels = list(metrics.keys())
        values = list(metrics.values())

        # Plot metrics
        fig, ax = plt.subplots()
        ax.barh(labels, values, color="skyblue")
        ax.set_xlabel("Similarity Scores")
        ax.set_title(f"Similarity Metrics for Article {idx}")
        st.pyplot(fig)

    st.write("---")