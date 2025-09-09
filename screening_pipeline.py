import torch
import pickle
from models.classifier import BERTResumeClassifier
from utils.dataset import ResumeDataset
from utils.similarity_utils import get_bert_embeddings, compute_similarity
import pandas as pd

# Load trained model and label encoder (do this once at module level)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTResumeClassifier(n_classes=24)
model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_resume(text):
    """Predict resume category and confidence score"""
    dataset = ResumeDataset([text], [0])
    batch = dataset[0]
    
    with torch.no_grad():
        input_ids = batch['input_ids'].unsqueeze(0).to(device)
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
        
        outputs = model(input_ids, attention_mask)
        predicted = torch.argmax(outputs, dim=1)
        
        category = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        confidence = torch.softmax(outputs, dim=1).max().item()
    
    return category, confidence

def screen_resume(resume_text, job_description):
    """Complete resume screening function"""
    # Step 1: Classify resume category
    category, cat_confidence = predict_resume(resume_text)
    
    # Step 2: Calculate job similarity
    resume_emb = get_bert_embeddings([resume_text])
    job_emb = get_bert_embeddings([job_description])
    similarity = compute_similarity(resume_emb[0], job_emb[0]).item()
    
    # Step 3: Overall screening score
    screening_score = (cat_confidence * 0.4) + (similarity * 0.6)
    
    return {
        'category': category,
        'category_confidence': cat_confidence,
        'job_similarity': similarity,
        'screening_score': screening_score
    }

def main():
    """Example usage of the screening pipeline"""
    # Example resume text
    sample_resume = """
    Experienced data scientist with 5 years of expertise in Python, machine learning, 
    and deep learning. Proficient in TensorFlow, PyTorch, pandas, and scikit-learn. 
    Strong background in statistical analysis and data visualization.
    """
    
    # Example job description
    sample_job = """
    We are looking for a Data Scientist with experience in machine learning, 
    Python programming, and statistical analysis. Knowledge of deep learning 
    frameworks like TensorFlow or PyTorch is preferred.
    """
    
    # Screen the resume
    result = screen_resume(sample_resume, sample_job)
    
    print("=== Resume Screening Results ===")
    print(f"Predicted Category: {result['category']}")
    print(f"Category Confidence: {result['category_confidence']:.2%}")
    print(f"Job Similarity: {result['job_similarity']:.3f}")
    print(f"Overall Screening Score: {result['screening_score']:.3f}")
    
    # Interpretation
    if result['screening_score'] >= 0.8:
        print("STRONG MATCH - Highly recommended candidate")
    elif result['screening_score'] >= 0.6:
        print("GOOD MATCH - Recommended for further review")
    else:
        print(" WEAK MATCH - Not recommended")

if __name__ == "__main__":
    main()
