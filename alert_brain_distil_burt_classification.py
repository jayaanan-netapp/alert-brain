import re

import openpyxl
from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizerFast, \
     DataCollatorWithPadding, pipeline
from datasets import Dataset, metric
import numpy as np
import evaluate

def compute_metrics(eval_pred):  # custom method to take in logits and calculate accuracy of the eval set
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# simple function to batch tokenize utterances with truncation
def preprocess_function(examples):
    return tokenizer(examples["utterance"], truncation=True)

wb = openpyxl.load_workbook('C:/Users/jayaanan/Downloads/Alerts_Data1.xlsx')
# Get the second sheet
sheet = wb['Sheet2']
# This code segment parses the snips dataset into a more manageable format

utterances = []
sequence_labels = []

for row in sheet.iter_rows(values_only=True):
    # Convert any None values to NULL
    row = [value if value is not None else 'NULL' for value in row]
    alerts = row[1].splitlines()
    # print(alerts)
    # Using a for loop
    for string in alerts:
        alert = re.sub(r"\b\d+\.", "", string, count=1)
        utterances.append(alert)
        sequence_labels.append(row[0])

print('length of all tokens', len(utterances), len(sequence_labels))
print('first utterance', utterances[0], 'sequence label',sequence_labels[0])
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
unique_sequence_labels = list(set(sequence_labels))
print('unique_sequence_labels-->',unique_sequence_labels)
sequence_labels = [unique_sequence_labels.index(l) for l in sequence_labels]
print(f'There are {len(unique_sequence_labels)} unique sequence labels')

print('utterances', utterances[0], len(utterances))
print('sequence_labels', sequence_labels[0], len(sequence_labels))
print(unique_sequence_labels[sequence_labels[0]])

snips_dataset = Dataset.from_dict(
    dict(
        utterance=utterances,
        label=sequence_labels
    )
)

snips_dataset = snips_dataset.train_test_split(test_size=0.2)
print(snips_dataset)

print(tokenizer('hi'))
print(tokenizer.decode([101, 2603, 1142, 18977, 126, 2940, 102]))


seq_clf_tokenized_snips = snips_dataset.map(preprocess_function, batched=True)



# only input_ids, attention_mask, and label are used. The rest are for show
print(seq_clf_tokenized_snips['train'][0])

# DataCollatorWithPadding creates batch of data. It also dynamically pads text to the
#  length of the longest element in the batch, making them all the same length.
#  It's possible to pad your text in the tokenizer function with padding=True, dynamic padding is more efficient.

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print({i: l for i, l in enumerate(unique_sequence_labels)})

sequence_clf_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-cased',
    num_labels=len(unique_sequence_labels),
)

# set an index -> label dictionary
sequence_clf_model.config.id2label = {i: l for i, l in enumerate(unique_sequence_labels)}
print(sequence_clf_model.config.id2label[0])
metric = evaluate.load("accuracy")

epochs = 5

training_args = TrainingArguments(
    output_dir="./alert_brain/results",
    num_train_epochs=epochs,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    load_best_model_at_end=True,

    # some deep learning parameters that the Trainer is able to take in
    warmup_steps=len(seq_clf_tokenized_snips['train']) // 5,  # number of warmup steps for learning rate scheduler,
    weight_decay=0.05,

    logging_steps=1,
    log_level='info',
    evaluation_strategy='epoch',
    eval_steps=50,
    save_strategy='epoch'
)

# Define the trainer:

trainer = Trainer(
    model=sequence_clf_model,
    args=training_args,
    train_dataset=seq_clf_tokenized_snips['train'],
    eval_dataset=seq_clf_tokenized_snips['test'],
    compute_metrics=compute_metrics,  # optional
    data_collator=data_collator
)
print(trainer.evaluate())
trainer.train()
print(trainer.evaluate())
trainer.save_model()
# We can now load our fine-tuned from our directory
pipe = pipeline("text-classification", "./alert_brain/results", tokenizer=tokenizer)

for index, utterance in enumerate(utterances):
    print(utterance, pipe(utterance), sequence_labels[index])
