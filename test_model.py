import torch
import pickle
from models.classifier import BERTResumeClassifier
from utils.dataset import ResumeDataset
import pandas as pd

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTResumeClassifier(n_classes=24)
model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_resume(text):
    """Predict resume category and confidence"""
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

def main():
    # Test with sample resume
    sample_resume = "Experienced data scientist with Python, machine learning, and deep learning expertise..."
    category, confidence = predict_resume(sample_resume)
    
    print(f"Predicted Category: {category}")
    print(f"Confidence: {confidence:.2%}")
    
    # Test with resume from dataset
    try:
        resume_df = pd.read_csv('data/resume_texts.csv')
        test_resume = resume_df.iloc[0]['resume_text']
        actual_category = resume_df.iloc[0]['category']
        
        predicted_category, confidence = predict_resume(test_resume)
        
        print(f"\n=== Real Resume Test ===")
        print(f"Actual Category: {actual_category}")
        print(f"Predicted Category: {predicted_category}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Match: {'✅' if actual_category == predicted_category else '❌'}")
        
    except FileNotFoundError:
        print("Resume dataset not found. Create data/resume_texts.csv first.")

if __name__ == "__main__":
    main()
