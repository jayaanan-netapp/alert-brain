import re

import openpyxl
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix

# Load BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('distilbert-base-cased')
tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased')
df = pd.DataFrame(columns=['category', 'message'], dtype=object)
# Load CSV file
wb = openpyxl.load_workbook('C:/Users/jayaanan/Downloads/Alerts_Data1.xlsx')
# Get the second sheet
sheet = wb['Sheet2']
# Iterate over rows in the sheet
for row in sheet.iter_rows(values_only=True):
    # Convert any None values to NULL
    row = [value if value is not None else 'NULL' for value in row]
    alerts = row[1].splitlines()
    # print(alerts)
    # Using a for loop
    for string in alerts:
        alert = re.sub(r"\b\d+\.", "", string, count=1)
        df = df.append({'category': row[0], 'message': alert}, ignore_index=True)
print(df)

# Create a dictionary to store data for each category
category_data = {}

# Group data by category
grouped_data = df.groupby('category')

# Iterate over each category and randomly select data for training
for category, group in grouped_data:
    indices = group.index.tolist()
    selected_indices = np.array(indices[:33])  # Convert to numpy array
    np.random.shuffle(selected_indices)  # Shuffle the indices
    category_data[category] = group.loc[selected_indices.tolist()]  # Convert back to list

# Concatenate the data from different categories
selected_data = pd.concat(category_data.values(), ignore_index=True)

# Shuffle the selected data
selected_data = selected_data.sample(frac=1).reset_index(drop=True)

# Tokenize alert messages
input_ids = []
attention_masks = []

for message in selected_data['message']:
    encoded_message = tokenizer.encode_plus(
        message,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    input_ids.append(encoded_message['input_ids'])
    attention_masks.append(encoded_message['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)

# Define BERT-based classification model
input_ids_input = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
attention_mask_input = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
bert_output = bert_model(input_ids_input, attention_mask=attention_mask_input)[1]  # Use pooled output
output = tf.keras.layers.Dense(5, activation='softmax')(bert_output)  # 5 categories

model = tf.keras.models.Model(inputs=[input_ids_input, attention_mask_input], outputs=output)

# Compile and train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare the training labels
labels = pd.Categorical(selected_data['category'])
labels = labels.codes  # Convert categories to numerical codes

print('labels', labels)
# Convert tensors to numpy arrays
input_ids_np = input_ids.numpy()
attention_masks_np = attention_masks.numpy()
labels_np = labels

# Split the data into training and evaluation sets
train_input_ids, eval_input_ids, train_attention_masks, eval_attention_masks, train_labels, eval_labels = \
    train_test_split(input_ids_np, attention_masks_np, labels_np, test_size=0.33, random_state=42)

# Train the model
model.fit(
    x=[train_input_ids, train_attention_masks],
    y=train_labels,
    batch_size=32,
    epochs=5
)

# Evaluate the model
eval_predictions = model.predict([eval_input_ids, eval_attention_masks])
eval_predicted_labels = np.argmax(eval_predictions, axis=1)

# Map labels to categories
category_map = {
    0: 'Connector',
    1: 'Performance',
    2: 'Costing',
    3: 'SaaSInfra',
    4: 'Miscellaneous'
}

eval_predicted_categories = [category_map[label] for label in eval_predicted_labels]
eval_true_categories = [category_map[label] for label in eval_labels]

# Calculate accuracy
accuracy = np.sum(np.array(eval_predicted_categories) == np.array(eval_true_categories)) / len(eval_true_categories)
print("Accuracy: {:.2%}".format(accuracy))

# Generate classification report
print(classification_report(eval_true_categories, eval_predicted_categories))

# Generate confusion matrix
confusion_mat = confusion_matrix(eval_true_categories, eval_predicted_categories)
print("Confusion Matrix:")
print(confusion_mat)
