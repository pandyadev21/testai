from flask import Flask, render_template, request
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the data
df1 = pd.read_excel('laws.xlsx')
df2 = pd.read_excel('laws2.xlsx')
# Add more dataframes for additional databases (laws3.xlsx, laws4.xlsx, etc.)

# Preprocess Data
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

df1['processed_description'] = df1['Description'].apply(preprocess_text)
df2['processed_description'] = df2['Description'].apply(preprocess_text)
# Apply the same preprocessing steps for additional dataframes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    user_input = request.form['incident']
    processed_user_input = preprocess_text(user_input)

    selected_databases = []
    for cb in ['CB1', 'CB2', 'CB3', 'CB4']:
        if cb in request.form:
            selected_databases.append(cb)

    all_descriptions = []
    relevant_databases = []  # Keep track of which databases were selected

    for db in selected_databases:
        if db == 'CB1':
            all_descriptions.extend(df1['processed_description'].tolist())
            relevant_databases.append(df1)
        elif db == 'CB2':
            all_descriptions.extend(df2['processed_description'].tolist())
            relevant_databases.append(df2)
        # Add similar conditions for additional databases

    print("Selected databases:", selected_databases)
    print("Relevant databases:", relevant_databases)

    if not relevant_databases:
        return "No relevant databases selected"

    all_descriptions.append(processed_user_input)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_descriptions)

    tfidf_matrix_train = tfidf_matrix[:-1].toarray()
    tfidf_matrix_user = tfidf_matrix[-1].toarray()

    cosine_similarities = cosine_similarity(tfidf_matrix_train, tfidf_matrix_user)
    most_similar_index = cosine_similarities.argmax()

    print("Most similar index:", most_similar_index)
    print("Length of relevant databases:", len(relevant_databases))

    if relevant_databases:
        most_similar_index = min(most_similar_index, len(relevant_databases[0]) - 1)
        relevant_df = relevant_databases[0].iloc[most_similar_index]
    else:
        relevant_df = None

    if relevant_df is not None and not relevant_df.empty:
        relevant_ipc_section = relevant_df['IPC Section']
        relevant_description = relevant_df['Description']
    else:
        relevant_ipc_section = "No relevant IPC section found"
        relevant_description = "No relevant description found"

    return render_template('result.html', ipc_section=relevant_ipc_section, description=relevant_description)

if __name__ == '__main__':
    app.run(debug=True)
