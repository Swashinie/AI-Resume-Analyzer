import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from test_model import predict_resume

def evaluate_model():
    # Load test data
    resume_df = pd.read_csv('data/resume_texts.csv')
    
    # Use a sample for evaluation
    sample_size = 100
    test_resumes = resume_df.sample(sample_size, random_state=42)
    
    predictions = []
    true_labels = []
    
    print("Evaluating model performance...")
    
    for _, row in test_resumes.iterrows():
        predicted_category, confidence = predict_resume(row['resume_text'])
        predictions.append(predicted_category)
        true_labels.append(row['category'])
    
    # Generate classification report
    report = classification_report(true_labels, predictions)
    
    print("=== Model Performance Report ===")
    print(report)
    
    # Save results
    with open('data/model_evaluation.txt', 'w') as f:
        f.write(report)
    
    print(f"\nEvaluation complete! Report saved to: data/model_evaluation.txt")

if __name__ == "__main__":
    evaluate_model()
