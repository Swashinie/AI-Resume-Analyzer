import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from utils.dataset import ResumeDataset
from models.classifier import BERTResumeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pickle

def train():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load extracted resume data from CSV
    print("Loading resume data...")
    resume_df = pd.read_csv('data/resume_texts.csv')
    resumes = resume_df['resume_text'].tolist()
    categories = resume_df['category'].tolist()
    
    print(f"Loaded {len(resumes)} resumes from {len(set(categories))} categories")
    print(f"Categories: {sorted(set(categories))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(categories)

    # Split data with stratify
    X_train, X_val, y_train, y_val = train_test_split(
        resumes,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create datasets
    train_dataset = ResumeDataset(X_train, y_train)
    val_dataset = ResumeDataset(X_val, y_val) 

    # Data loaders - optimized batch size for GPU
    batch_size = 16 if torch.cuda.is_available() else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = BERTResumeClassifier(n_classes=len(label_encoder.classes_))
    model.to(device)

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training parameters
    epochs = 5
    best_val_loss = float('inf')

    print("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_batches += 1

        # Calculate average losses
        avg_train_loss = total_train_loss / train_batches
        avg_val_loss = total_val_loss / val_batches
        
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f} - Saving model...")
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save model state
            torch.save(model.state_dict(), "models/best_model.pth")
            
            # Save label encoder
            with open("models/label_encoder.pkl", "wb") as f:
                pickle.dump(label_encoder, f)

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved with {len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}")
    print("Files saved:")
    print("- models/best_model.pth")
    print("- models/label_encoder.pkl")

if __name__ == "__main__":
    train()
