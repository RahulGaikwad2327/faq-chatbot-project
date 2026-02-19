from flask import Flask, render_template, request, jsonify
import json
import nltk
import math
from collections import Counter

nltk.download('punkt')

app = Flask(__name__)

with open("faqs.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    return tokens

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    user_tokens = preprocess(user_input)
    user_vector = Counter(user_tokens)

    similarities = []

    for question in questions:
        question_tokens = preprocess(question)
        question_vector = Counter(question_tokens)
        sim = cosine_similarity(user_vector, question_vector)
        similarities.append(sim)

    best_match_index = similarities.index(max(similarities))
    response = answers[best_match_index]

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
