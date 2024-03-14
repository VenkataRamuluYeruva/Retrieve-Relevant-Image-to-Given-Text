'''
import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
from PIL import Image
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, url_for


app=Flask(__name__)


def text_to_vector(text):
    words = word_tokenize(text)

    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]

    processed_text = ' '.join(filtered_words)
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform([processed_text])

    return text_vector

def image_to_vector(image, num_features):
    image = Image.open(image)
    image_array = np.array(image).flatten()
    reduced_image_vector = image_array[:num_features]
    x=reduced_image_vector.reshape(1, -1)
    return x


def calculate_cosine_similarity(text_vector, image_vector):
    return cosine_similarity(text_vector, image_vector)[0][0]

def getImage(comments):
    df = pd.read_csv("./results.csv", sep='|')
    df.columns = ['image_name', 'comment_number', 'comment']
    df['comment'] = df['comment'].str.replace(',', '')

    input_comments = comments

    images = []
    for input_comment in input_comments:
        for i, row in df.iterrows():
            if input_comment.strip() == str(row['comment']).strip():
                images.append([input_comment, row['image_name']])
                break

    image_path = './static/Images'
    Similarity_Scores = []

    for image in images:
        text_vector = text_to_vector(image[0])
        image_vector = image_to_vector(f"{image_path}/{image[1]}", text_vector.shape[1])
        similarity_score = calculate_cosine_similarity(text_vector, image_vector)
        Similarity_Scores.append([image[1], similarity_score, image[0]])

    return Similarity_Scores


@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        comment=list(request.form['comment'].split(","))
        results = getImage(comment)
        return render_template('index.html', results=results)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
'''
# app.py
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

from flask import Flask, render_template, request
from comment_similarity_model import CommentSimilarityModel

app = Flask(__name__)
model = CommentSimilarityModel("./results.csv")
image_path = './static/Images'

def image_to_vector(image, num_features):   
    image = Image.open(image)
    image_array = np.array(image).flatten()
    reduced_image_vector = image_array[:num_features]
    x=reduced_image_vector.reshape(1, -1)
    return x

def text_to_vector(text):
    words = word_tokenize(text)

    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]

    processed_text = ' '.join(filtered_words)
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform([processed_text])

    return text_vector

def calculate_cosine_similarity(text_vector, image_vector):
    return cosine_similarity(text_vector, image_vector)[0][0]

@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        input_comments = request.form.get('comment')
        input_comments = input_comments.split(',')
        input_comments = [comment.strip() for comment in input_comments]

        for input_comment in input_comments:
            comment, image_name, similarity = model.get_most_similar_comment(input_comment)
            Similarity_Scores = []
            text_vector = text_to_vector(input_comment)
            image_vector = image_to_vector(f"{image_path}/{image_name}", text_vector.shape[1])
            similarity_score = calculate_cosine_similarity(text_vector, image_vector)
            Similarity_Scores.append([image_name, similarity_score,input_comment])
        print(Similarity_Scores)

        return render_template('index.html', results=Similarity_Scores)
           

    else: 
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

