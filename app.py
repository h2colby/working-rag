from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rag_pipeline import answer_question

app = Flask(__name__)

# Configure CORS with specific options
CORS(app, resources={
    r"/ask": {
        "methods": ["POST", "OPTIONS", "HEAD"],
        "allow_headers": ["Content-Type"]
    },
    r"/test": {
        "origins": ["http://localhost:3000", "http://192.168.2.232:3000"],
        "methods": ["GET", "OPTIONS"]
    }
})

@app.route('/ask', methods=['HEAD'])
def handle_head():
    response = app.make_default_options_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.get_json()
    question = data.get('question', '')
    print(f"Received question: {question}")  # Log the input question

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Call the updated answer_question to get both context and response
        result = answer_question(question)
        print(f"Result from answer_question: {result}")  # Log the result dictionary

        # Return the result directly without transforming the key
        return jsonify(result)
    except Exception as e:
        print(f"Error occurred: {e}")  # Log the exception details
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)