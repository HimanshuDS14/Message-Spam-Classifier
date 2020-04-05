from flask import Flask , render_template , url_for , request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__ , static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict' , methods = ["POST" , "GET"])
def predict():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)

    data['label'] = data["class"].map({'ham': 0, 'spam': 1})
    data.drop('class', axis=1, inplace=True)

    lemma = WordNetLemmatizer()
    corpus = []

    for i in range(0, len(data)):
        words = re.sub('[^a-zA-Z]', ' ', data["message"][i])
        words = words.lower()
        words = words.split()

        words = [lemma.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        words = ' '.join(words)

        corpus.append(words)



    cv = TfidfVectorizer(max_features=1500)
    x = cv.fit_transform(corpus).toarray()

    y = data["label"]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

    m = MultinomialNB()
    m.fit(train_x, train_y)


    if request.method == 'POST':
        message = request.form['message']
        data = [message]

        vect = cv.transform(data).toarray()

        my_prediction = m.predict(vect)

        if my_prediction == [0]:
            my_prediction = "Not a Spam"
        else:
            my_prediction = "Spam"



    
    return render_template('index.html' , prediction_text = "Your Message is {}".format(my_prediction))

if __name__ == '__main__':
    app.run(debug = True)
    
    
