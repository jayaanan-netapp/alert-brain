import openpyxl
import re
import pandas as pd
import mysql.connector
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
import re

import openpyxl
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix

'''CREATE TABLE `alerts` (
	`id` INT(10) NOT NULL AUTO_INCREMENT,
	`message` MEDIUMTEXT NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`category` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`gpt_category` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`svm_category` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`nb_category` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`lda_category` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`nmf_category` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`bert_category` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
	INDEX `Index 1` (`id`) USING BTREE
)
COLLATE='utf8mb4_0900_ai_ci'
ENGINE=InnoDB
;
'''

# Modify the display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.width', None)  # Auto-adjust the display width


class AlertBrain:

    def __init__(self, host, user, password, database):
        self.df = pd.DataFrame(columns=['category', 'message'], dtype=object)
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database)
        self.conn.connect()
        self.cursor = self.conn.cursor()

    def close(self):
        self.cursor.close()
        if self.conn.is_connected():
            self.conn.close()
            print("Connection closed")

    def load_and_clean_data(self, file_path, sheet_name):
        # Open the Excel file
        wb = openpyxl.load_workbook(file_path)
        # Get the second sheet
        sheet = wb[sheet_name]
        # Iterate over rows in the sheet
        for row in sheet.iter_rows(values_only=True):
            # Convert any None values to NULL
            row = [value if value is not None else 'NULL' for value in row]
            alerts = row[1].splitlines()
            # print(alerts)
            # Using a for loop
            for string in alerts:
                alert = re.sub(r"\b\d+\.", "", string, count=1)
                # print(row[0], alert)
                query = "INSERT INTO alerts (category, message) VALUES (%s, %s)"
                values = (row[0], alert)
                self.df = self.df.append({'category': row[0], 'message': alert}, ignore_index=True)

                # Executing the query with the values
                self.cursor.execute(query, values)
                self.conn.commit()
        # print(df)
        # Check for the existence of NaN values in a cell:
        self.df.isnull().sum()
        self.df.dropna(inplace=True)
        # Delete and remove empty strings.
        blanks = []  # start with an empty list
        for i, lb, rv in self.df.itertuples():  # iterate over the DataFrame
            if type(rv) == str:  # avoid NaN values
                if rv.isspace():  # test 'review' for whitespace
                    blanks.append(i)  # add matching index numbers to the list
        # print(len(blanks), 'blanks: ', blanks)
        self.df.drop(blanks, inplace=True)

    def execute_supervised_learning_modal(self, text_clf_lsvc, column_name):

        X = self.df['message']
        y = self.df['category']
        '''
        Section for supervised learning. Hear we are evaluating NB and SVM to train and test on alerts.
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        text_clf_lsvc.fit(X_train, y_train)
        # Form a prediction set
        predictions = text_clf_lsvc.predict(X_test)
        # Get corresponding texts for the predictions
        predicted_texts = []
        for prediction, text in zip(predictions, X_test):
            predicted_texts.append((prediction, text))
        # Print the predictions and corresponding texts
        for prediction, text in predicted_texts:
            '''print("Prediction:", prediction)
            print("Text:", text)
            print()'''
            query = 'SELECT * FROM alerts WHERE message = %s'
            self.cursor.execute(query, (text,))
            rows = self.cursor.fetchall()
            for row in rows:
                query = f"UPDATE alerts SET {column_name} = %s WHERE id = %s"
                self.cursor.execute(query, (prediction, row[0]))
        # Update test data.
        for train in X_train:
            query = 'SELECT * FROM alerts WHERE message = %s'
            self.cursor.execute(query, (train,))
            rows = self.cursor.fetchall()
            for row in rows:
                query = f"UPDATE alerts SET {column_name} = %s WHERE id = %s"
                self.cursor.execute(query, (row[2], row[0]))
        # Commit the changes to the database
        print(metrics.confusion_matrix(y_test, predictions))
        # Print a classification report
        print(metrics.classification_report(y_test, predictions))
        # Print the overall accuracy
        print(metrics.accuracy_score(y_test, predictions))
        self.conn.commit()

    def un_supervise_learning(self):
        cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        dtm = cv.fit_transform(self.df['message'])
        LDA = LatentDirichletAllocation(n_components=5, random_state=42)
        LDA.fit(dtm)
        for index, topic in enumerate(LDA.components_):
            print(f'THE TOP 3 WORDS FOR TOPIC #{index}')
            print([cv.get_feature_names()[i] for i in topic.argsort()[-3:]])
        topic_results = LDA.transform(dtm)
        self.df["LDA_Topic"] = topic_results.argmax(axis=1)
        lda_topic_dictionary = {0: 'memory', 1: 'virtual_machine', 2: 'performance', 3: 'connection', 4: 'cloud'}
        self.df["LDA_Topic_label"] = self.df["LDA_Topic"].map(lda_topic_dictionary)
        print(self.df)

        tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        dtm = tfidf.fit_transform(self.df['message'])
        nmf_model = NMF(n_components=5, random_state=42)
        nmf_model.fit(dtm)
        for index, topic in enumerate(nmf_model.components_):
            print(f'THE TOP 3 WORDS FOR TOPIC #{index}')
            print([tfidf.get_feature_names()[i] for i in topic.argsort()[-3:]])
        topic_results = LDA.transform(dtm)
        self.df["NMF_Topic"] = topic_results.argmax(axis=1)
        lda_topic_dictionary = {0: 'service_alert', 1: 'hypervisor', 2: 'cloud', 3: 'data_inconsistent',
                                4: 'kubernetes'}
        self.df["NMF_Topic_label"] = self.df["NMF_Topic"].map(lda_topic_dictionary)
        print(self.df)
        # Iterate over the DataFrame rows and insert data related to unsupervised learning
        for row in self.df.iterrows():
            # Get the text from the row
            text = row[1]['message']
            query = 'SELECT * FROM alerts WHERE message = %s'
            self.cursor.execute(query, (text,))
            rows = self.cursor.fetchall()
            for data_row in rows:
                query = 'UPDATE alerts SET lda_category = %s WHERE id = %s'
                self.cursor.execute(query, (row[1]['LDA_Topic_label'], data_row[0]))
                query = 'UPDATE alerts SET nmf_category = %s WHERE id = %s'
                self.cursor.execute(query, (row[1]['NMF_Topic_label'], data_row[0]))
        self.conn.commit()

    def supervise_learning(self):
        # Naïve Bayes:
        text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                                ('clf', MultinomialNB()),
                                ])
        self.execute_supervised_learning_modal(text_clf_nb, 'nb_category')

        # Linear SVC:
        text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                                  ('clf', LinearSVC()),
                                  ])
        self.execute_supervised_learning_modal(text_clf_lsvc, 'svm_category')

    def deep_learning(self):
        # Load BERT model and tokenizer
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        category_data = {}

        # Group data by category
        grouped_data = self.df.groupby('category')

        '''for name_of_the_group, group in grouped_data:
            print(name_of_the_group)
            print(group)'''

        # Iterate over each category and randomly select data for training
        for category, group in grouped_data:
            indices = group.index.tolist()
            selected_indices = np.array(indices)  # Convert to numpy array
            np.random.shuffle(selected_indices)  # Shuffle the indices
            category_data[category] = group.loc[selected_indices.tolist()]  # Convert back to list

        # Concatenate the data from different categories
        selected_data = pd.concat(category_data.values(), ignore_index=True)
        # print('selected_data-->', selected_data)
        # Shuffle the selected data
        selected_data = selected_data.sample(frac=1).reset_index(drop=True)
        # print('Shuffle_data-->', selected_data)
        # Tokenize alert messages
        input_ids = []
        attention_masks = []

        for message in selected_data['message']:
            encoded_message = tokenizer.encode_plus(
                message,
                add_special_tokens=True,
                max_length=512,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='tf'
            )

            input_ids.append(encoded_message['input_ids'])
            attention_masks.append(encoded_message['attention_mask'])

        input_ids = tf.concat(input_ids, axis=0)
        attention_masks = tf.concat(attention_masks, axis=0)

        # Define BERT-based classification model
        input_ids_input = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)
        attention_mask_input = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)
        bert_output = bert_model(input_ids_input, attention_mask=attention_mask_input)[1]  # Use pooled output
        output = tf.keras.layers.Dense(5, activation='softmax')(bert_output)  # 5 categories

        model = tf.keras.models.Model(inputs=[input_ids_input, attention_mask_input], outputs=output)

        # Compile and train the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Prepare the training labels
        labels = pd.Categorical(selected_data['category'])
        labels = labels.codes  # Convert categories to numerical codes

        # Convert tensors to numpy arrays
        input_ids_np = input_ids.numpy()
        attention_masks_np = attention_masks.numpy()
        labels_np = labels

        # Split the data into training and evaluation sets
        train_input_ids, eval_input_ids, train_attention_masks, eval_attention_masks, train_labels, eval_labels = \
            train_test_split(input_ids_np, attention_masks_np, labels_np, test_size=0.33, random_state=42)
        # print('split data-->',train_input_ids, eval_input_ids, train_attention_masks, eval_attention_masks, train_labels, eval_labels  )
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
        # Convert eval_labels to Pandas Series
        eval_labels = pd.Series(eval_labels)

        eval_predicted_categories = [category_map[label] for label in eval_predicted_labels]
        eval_true_categories = [category_map[label] for label in eval_labels]

        eval_messages = selected_data.loc[eval_labels.index, 'message'].tolist()

        # Print test messages and predictions
        print("Test Messages:")
        for message, true_category, predicted_category in zip(eval_messages, eval_true_categories,
                                                              eval_predicted_categories):
            print("Message: ", message)
            print("True Category: ", true_category)
            print("Predicted Category: ", predicted_category)
            print("-----------------------------------------")
            query = 'SELECT * FROM alerts WHERE message = %s'
            self.cursor.execute(query, (message,))
            rows = self.cursor.fetchall()
            for data_row in rows:
                query = 'UPDATE alerts SET bert_category = %s WHERE id = %s'
                self.cursor.execute(query, (predicted_category, data_row[0]))
        self.conn.commit()
        train_labels_series = pd.Series(train_labels)
        train_messages = selected_data.loc[train_labels_series, 'message'].tolist()
        for train in train_messages:
            query = 'SELECT * FROM alerts WHERE message = %s'
            self.cursor.execute(query, (train,))
            rows = self.cursor.fetchall()
            for row in rows:
                query = f"UPDATE alerts SET bert_category = %s WHERE id = %s"
                self.cursor.execute(query, (row[2], row[0]))
        self.conn.commit()
        # Calculate accuracy
        accuracy = np.sum(np.array(eval_predicted_categories) == np.array(eval_true_categories)) / len(
            eval_true_categories)
        print("Accuracy: {:.2%}".format(accuracy))

        # Generate classification report
        print(classification_report(eval_true_categories, eval_predicted_categories))

        # Generate confusion matrix
        confusion_mat = confusion_matrix(eval_true_categories, eval_predicted_categories)
        print("Confusion Matrix:")
        print(confusion_mat)


if __name__ == '__main__':
    brain = AlertBrain("localhost", "root", "Netapp!2345", "alert_brain")
    brain.load_and_clean_data('C:/Users/jayaanan/Downloads/Alerts_Data1.xlsx', 'Sheet2')
    brain.supervise_learning()
    brain.un_supervise_learning()
    brain.deep_learning()
    brain.close()
