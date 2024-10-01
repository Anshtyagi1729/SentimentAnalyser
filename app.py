from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text')
    result = sentiment_pipeline(text)[0]
    return jsonify({"label": result['label'], "score": result['score']})

if __name__ == "__main__":
    app.run(debug=True)
