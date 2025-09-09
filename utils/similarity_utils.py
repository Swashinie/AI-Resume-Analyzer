import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    """Generate BERT embeddings for a list of texts"""
    embeddings = []
    
    for text in texts:
        # Tokenize and encode
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          padding=True, max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
    
    return np.array(embeddings)

def compute_similarity(emb1, emb2):
    """Compute cosine similarity between embeddings"""
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    
    return cosine_similarity(emb1, emb2)[0]

# Load data with correct paths
try:
    # Load resume data
    resume_df = pd.read_csv('data/resume_texts.csv')
    resume_texts = resume_df['resume_text'].tolist()
    
    # Load job descriptions
    job_desc_df = pd.read_csv('data/job_descriptions.csv') 
    job_descriptions = job_desc_df['cleaned_description'].tolist()
    
except FileNotFoundError as e:
    print(f"Warning: CSV files not found - {e}")
    resume_texts = []
    job_descriptions = []
