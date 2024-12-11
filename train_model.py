import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load and prepare the data
data = pd.read_csv('nl_to_sql.csv')
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['question'].tolist(), data['sql_query'].tolist(), test_size=0.1, random_state=42
)

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize the input texts
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")

# Tokenize the labels separately
train_labels_encodings = tokenizer(train_labels, padding=True, truncation=True, return_tensors="pt").input_ids
val_labels_encodings = tokenizer(val_labels, padding=True, truncation=True, return_tensors="pt").input_ids

# Custom dataset class
class NLSQLDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create train and validation datasets
train_dataset = NLSQLDataset(train_encodings, train_labels_encodings)
val_dataset = NLSQLDataset(val_encodings, val_labels_encodings)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
