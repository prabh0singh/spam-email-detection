# %% [markdown]
# # **Email spam Detection**
# Email spam detection system is used to detect email spam using Machine Learning technique called Natural Language Processing and Python,
# where we have a dataset contain a lot of emails by extract important words and then use naive classifier we can detect if this email is spam or not.

# %% [markdown]
# ### **Libraries**

# %%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
import string
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('stopwords')

# %% [markdown]
# ### **Project Pipeline**
# For any machine learning project it consist of three main phases as following :-
# 1. **Scoping :** List the problem description and project goals
# 2. **The Data :** Load, analyse and prepare the dataset for training
# 3. **The Model :** Create and train the model on the dataset

# %% [markdown]
# ## **1 | Scoping**
# The project problem is that we have a dataset containing a set of emails and we will use machine learning and nlp techniques in order to determine if this email is spam or not.

# %% [markdown]
# ## **2 | The Data**
# In this phase we will analyze and prepare the dataset before training by applying varoius steps as following :-
# 1. Data Loading
# 2. Data Visualization
# 3. Data Cleaning
# 4. Data Splitting

# %% [markdown]
# ##### **1. Data Loading**
# Load the dataset and show its info and statistics

# %%
# Load the dataset
dataset = pd.read_csv('dataset/emails.csv')
dataset.shape

# %%
# Show dataset head (first 5 records)
dataset.head() 

# %%
# Show dataset info
dataset.info()

# %%
# Show dataset statistics
dataset.describe()

# %% [markdown]
# ##### **2. Data Visualization**
# Visualize dataset features frequencies to get some insights

# %%
# Visualize spam  frequenices
plt.figure(dpi=100)
sns.countplot(dataset['spam'])
plt.title("Spam Freqencies")
plt.show()

# %% [markdown]
# ##### **3. Data Cleaning**
# Handling missing values and check for duplicates 

# %%
# Check for missing data for each column 
dataset.isnull().sum()

# %%
# Check for duplicates and remove them 
dataset.drop_duplicates(inplace=True)

# %%
# Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# %%
# Fit the CountVectorizer to data
message = CountVectorizer(analyzer=process).fit_transform(dataset['text'])

# %%
# Save the vectorizer
dump(message, open("models/vectorizer.pkl", "wb"))

# %% [markdown]
# ##### **4. Data Splitting**
# Split the dataset into training and testing sets

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.20, random_state=0)

# %% [markdown]
# ## **3. The Model**
# In this phase we will create and train a multinomial naive bayes model 

# %%
# Model creation
model = MultinomialNB()

# %%
# Model training
model.fit(X_train, y_train)

# %%
# Model saving
dump(model, open("models/model.pkl", 'wb'))

# %%
# Model predictions on test set
y_pred = model.predict(X_test)

# %%
# Model Evaluation | Accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy * 100

# %%
# Model Evaluation | Classification report
classification_report(y_test, y_pred)

# %%
# Model Evaluation | Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(dpi=100)
sns.heatmap(cm, annot=True)
plt.title("Confusion matrix")
plt.show()

# %% [markdown]
# 4. Spam vs Non-Spam Visualisation

# %%

import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('dataset/emails.csv')

# Calculate the number of spam and non-spam emails
spam_count = dataset['spam'].sum()
non_spam_count = len(dataset) - spam_count

# Create data for the pie chart
labels = ['Spam', 'Non-Spam']
sizes = [spam_count, non_spam_count]
colors = ['#ff9999', '#66b3ff']  # Colors for the pie chart slices

# Create a pie chart
plt.figure(dpi=100)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Spam vs Non-Spam Emails")

# Display the pie chart
plt.show()


