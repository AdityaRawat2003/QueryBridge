# QueryBridge
# Natural Language to SQL (NL-to-SQL) 

This project focuses on converting natural language queries into SQL statements using a fine-tuned Transformer-based model, specifically for disaster management scenarios. It includes data preprocessing, model training, and evaluation pipelines to enable efficient and accurate text-to-SQL translation.

---

## Overview
This project aims to bridge the gap between natural language and database queries by leveraging machine learning models. It focuses on:
- Simplifying database access for non-technical users.
- Streamlining query generation for disaster management systems.

The project fine-tunes a T5 model to translate natural language questions into SQL queries, ensuring high accuracy and relevance in disaster management contexts.

---

## Features
- **Data Preprocessing**: Converts raw datasets into clean and structured formats for model training.
- **Custom Dataset Support**: Easily integrate domain-specific datasets.
- **Fine-Tuning**: Leverages pre-trained T5 for high-quality SQL query generation.
- **Evaluation Metrics**: Includes accuracy, BLEU, and semantic similarity for robust evaluation.
- **Extensibility**: Modular design to support various datasets and domains.

---

## Installation
### Prerequisites
- Python >= 3.8
- PyTorch
- Hugging Face Transformers

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd nl-to-sql
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. **Data Preprocessing**:
   Use `preprocess_data.py` to clean and format raw datasets.
   ```bash
   python preprocess_data.py --input dataset.csv --output preprocessed_data.csv
   ```
2. **Model Training**:
   Train the model with `train_model.py`.
   ```bash
   python train_model.py --data preprocessed_data.csv --output model_dir
   ```
3. **Evaluation**:
   Evaluate the trained model using `model_evaluation.py`.
   ```bash
   python model_evaluation.py --model model_dir --data preprocessed_data.csv
   ```
   
---

## Datasets
- **Raw Dataset**: Contains natural language questions and corresponding SQL queries.
- **Preprocessed Dataset**: Cleaned and formatted for model training, structured with `input_text` and `target_text` columns.

---

## Model Training
1. Ensure the preprocessed dataset is ready.
2. Use `fine_tune_t5.py` for custom fine-tuning.
   ```bash
   python fine_tune_t5.py --train preprocessed_data.csv --output model_dir
   ```
3. Monitor training metrics for convergence.

---

## Sample
![WhatsApp Image 2024-11-22 at 14 04 16_52693200](https://github.com/user-attachments/assets/18a13e3d-b72d-470d-aff8-049be036505f)

![WhatsApp Image 2024-11-22 at 14 05 22_67b57715](https://github.com/user-attachments/assets/aa8ec998-c64d-487c-8bcf-539bf4913e1c)



