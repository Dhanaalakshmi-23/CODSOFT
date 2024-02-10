import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the SMS Spam Collection dataset
def load_dataset(file_path):
    messages = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                messages.append(parts[1])
                labels.append(parts[0])
    return messages, labels

# Preprocess the text data
def preprocess_text(messages):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    preprocessed = []
    for message in messages:
        # Tokenization
        words = word_tokenize(message.lower())
        # Removing punctuation and stop words, and stemming
        processed = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        preprocessed.append(' '.join(processed))
    return preprocessed

# Vectorize the preprocessed text data
def vectorize_text(train_data, test_data):
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
    return train_vectors, test_vectors

# Load dataset
messages, labels = load_dataset('sms_spam_collection.txt')

# Preprocess text
preprocessed_messages = preprocess_text(messages)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_messages, labels, test_size=0.2, random_state=42)

# Vectorize text
X_train_vectors, X_test_vectors = vectorize_text(X_train, X_test)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Predictions
predictions = classifier.predict(X_test_vectors)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))
