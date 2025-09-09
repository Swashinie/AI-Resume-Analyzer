import torch
from transformers import BertTokenizer
from models.classifier import BERTResumeClassifier
import pickle

def load_model_and_label_encoder(model_path="models/best_model.pth", le_path="models/label_encoder.pkl"):
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    model = BERTResumeClassifier(n_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, label_encoder

def predict_resume_category(resume_text, model, label_encoder):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(resume_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, confidence

if __name__ == "__main__":
    model, label_encoder = load_model_and_label_encoder()
    sample_resume = "Experienced software developer with expertise in Python, Java, and cloud computing."
    prediction, confidence = predict_resume_category(sample_resume, model, label_encoder)
    print(f"Predicted Category: {prediction}, Confidence: {confidence:.4f}")
