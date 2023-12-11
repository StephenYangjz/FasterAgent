from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, OPTForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import torch

# set seed
seed_value = 42
torch.manual_seed(seed_value)

# Check if a GPU is available and set it as the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your dataset
# Specify the paths to your training and test data files
train_file = '/home/zinccat/datasets/data/data/toolllama_G123_dfs_train_tokenlength_legal.json'
test_file = '/home/zinccat/datasets/data/data/toolllama_G123_dfs_eval_tokenlength_legal.json'

# Load the datasets
# Replace 'data_field_name' with the appropriate field name in your JSON structure
dataset = load_dataset('json', data_files={'train': train_file, 'test': test_file})
# Choose a model and tokenizer
model_name = "distilbert-base-uncased" #"facebook/opt-125m" # #allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_label(examples):
    tokenized_inputs = tokenizer([x.lstrip('\n').rstrip('\n') for x in examples['userprompt']], padding='max_length', truncation=True) #, max_length=2048)
    max_len = max([len(x) for x in tokenized_inputs['input_ids']])
    tokenized_inputs['labels'] = [float(x)/2048 for x in examples['num_tokens']] #[float(x)/5 for x in examples['num_api_calls']]
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_label, batched=True)

# Assuming 'num_api_calls' is the name of your target column
# Extract the 'num_api_calls' column and find the number of unique values
num_api_calls = dataset['train']['num_api_calls']
num_classes = len(np.unique(num_api_calls))

print(f"Number of classes: {num_classes}")

# Load model and send it to the GPU
# model = OPTForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
# for name, param in model.named_parameters():
#     if 'classifier' not in name:  # Freeze parameters not in the classifier
#         param.requires_grad = False

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32, #64,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_dir="./logs",
    logging_steps=200,
    save_steps=200,
    save_total_limit=1,
    # load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    gradient_accumulation_steps=1, #4,
    fp16=True
)

# Define the compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.01]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

# Create a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics_for_regression,
)

# Train the model
trainer.train()

# Evaluate the model
# evaluation_results = trainer.evaluate()

# # Print the accuracy
# print("Accuracy:", evaluation_results["eval_accuracy"])

# Save the model
trainer.save_model("./results")