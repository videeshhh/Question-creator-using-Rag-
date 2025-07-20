import os
import logging
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from rag_model import RAGQuestionAnsweringSystem

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this')

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rag_system = RAGQuestionAnsweringSystem(GOOGLE_API_KEY)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_or_create_session():
    if 'session_id' not in session:
        session['session_id'] = rag_system.create_session()
        logger.info(f"New session created: {session['session_id']}")
    return session['session_id']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "active_sessions": len(rag_system.list_active_sessions())
    }), 200

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400

        file = request.files['pdf']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File must be a PDF"}), 400

        session_id = get_or_create_session()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)

        if rag_system.process_pdf(filepath, session_id, filename):
            return jsonify({
                "message": "PDF processed successfully",
                "session_id": session_id,
                "filename": filename
            }), 200
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Failed to process PDF"}), 500

    except Exception as e:
        logger.error(f"upload_pdf error: {str(e)}")
        return jsonify({"error": f"Upload error: {str(e)}"}), 500

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        data = request.get_json() or {}
        num_questions = min(max(data.get('num_questions', 5), 1), 10)
        questions = rag_system.generate_questions(session_id, num_questions)

        if questions:
            return jsonify({
                "questions": questions,
                "session_id": session_id,
                "count": len(questions)
            }), 200
        return jsonify({"error": "Failed to generate questions"}), 500

    except Exception as e:
        logger.error(f"generate_questions error: {str(e)}")
        return jsonify({"error": f"Question generation error: {str(e)}"}), 500

@app.route('/api/get-answer', methods=['POST'])
def get_answer():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        result = rag_system.get_answer(session_id, question)

        if "error" in result:
            return jsonify(result), 500

        return jsonify({
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", "medium"),
            "session_id": session_id
        }), 200

    except Exception as e:
        logger.error(f"get_answer error: {str(e)}")
        return jsonify({"error": f"Answer generation error: {str(e)}"}), 500

@app.route('/api/questions', methods=['GET'])
def get_questions():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        questions = rag_system.get_questions_for_session(session_id)
        info = rag_system.get_session_info(session_id)

        return jsonify({
            "questions": questions,
            "session_id": session_id,
            "session_info": info,
            "count": len(questions)
        }), 200

    except Exception as e:
        logger.error(f"get_questions error: {str(e)}")
        return jsonify({"error": f"Error retrieving questions: {str(e)}"}), 500

@app.route('/api/session-info', methods=['GET'])
def get_session_info():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        info = rag_system.get_session_info(session_id)
        if not info:
            return jsonify({"error": "Session not found"}), 404

        return jsonify({
            "session_id": session_id,
            "session_info": info
        }), 200

    except Exception as e:
        logger.error(f"get_session_info error: {str(e)}")
        return jsonify({"error": f"Error retrieving session info: {str(e)}"}), 500

@app.route('/api/cleanup-session', methods=['POST'])
def cleanup_session():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        if rag_system.cleanup_session(session_id):
            session.clear()
            return jsonify({"message": "Session cleaned up successfully"}), 200
        return jsonify({"error": "Failed to cleanup session"}), 500

    except Exception as e:
        logger.error(f"cleanup_session error: {str(e)}")
        return jsonify({"error": f"Cleanup error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Max size is 16MB."}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
