from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



def semantic_search_with_sentiment(query, csv_rows,
                                   embedding_model,
                                   sentiment_model, sentiment_tokenizer):
    # Embed CSV rows and query
    embeddings = embedding_model.encode(csv_rows)
    query_embedding = embedding_model.encode([query])

    # Perform similarity search
    similarities = cosine_similarity(query_embedding, embeddings)
    best_match_idx = similarities.argmax()
    best_match_row = csv_rows[best_match_idx]

    # Analyze sentiment of the matched row
    inputs = sentiment_tokenizer(best_match_row, return_tensors="pt")
    outputs = sentiment_model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    sentiment_label = ["negative", "neutral", "positive"][sentiment]

    return {"match": best_match_row, "sentiment": sentiment_label}

# Example usage
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

csv_rows = [
    "The product is amazing and works perfectly.",
    "هذا المنتج سيء للغاية وغير مفيد.",
]

query = "Product performance"
result = semantic_search_with_sentiment(query, csv_rows, embedding_model, sentiment_model, sentiment_tokenizer)
print(result)
