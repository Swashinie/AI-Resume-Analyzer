from torch.utils.data import Dataset
from transformers import BertTokenizer

class ResumeDataset(Dataset):
    def __init__(self, resumes, labels, max_length=512):
        self.resumes = resumes
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.resumes)

    def __getitem__(self, idx):
        resume_text = self.resumes[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            resume_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
