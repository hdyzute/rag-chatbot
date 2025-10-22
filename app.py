import os
import uuid
import time
import tempfile
import logging
from threading import Thread
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings

# =========================
# 🔧 CẤU HÌNH & KHỞI TẠO
# =========================
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

SESSION_TIMEOUT = 3600
CHROMA_DIR = "./chroma_dbs"
os.makedirs(CHROMA_DIR, exist_ok=True)

SESSIONS = {}

# Lấy Ollama base URL từ environment
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# =========================
# 📌 TẢI MODEL MỘT LẦN
# =========================
try:
    logging.info("🚀 Đang kết nối Ollama Embeddings...")
    EMBEDDINGS_MODEL = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text"  # Model embedding nhỏ gọn
    )
    logging.info("✅ Embedding model sẵn sàng.")

    logging.info(f"🚀 Đang kết nối Ollama LLM tại {OLLAMA_BASE_URL}...")
    LLM = Ollama(model="llama3", temperature=0.3, base_url=OLLAMA_BASE_URL)
    LLM.invoke("ping")
    logging.info("✅ LLM sẵn sàng.")
except Exception as e:
    logging.error(f"❌ Lỗi tải model: {e}")
    EMBEDDINGS_MODEL = None
    LLM = None

# =========================
# 🧠 CLASS CHATBOT
# =========================
class RAGChatbot:
    def __init__(self, embeddings_model, llm, session_id):
        self.embeddings = embeddings_model
        self.llm = llm
        self.vectorstore = None
        self.qa_chain = None
        self.session_id = session_id
        self.last_access = time.time()
        logging.info(f"🔸 Khởi tạo chatbot cho session {session_id}")

    def load_and_index(self, file_path):
        try:
            # --- Load tài liệu ---
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError("❌ Chỉ hỗ trợ file .pdf hoặc .txt")

            documents = loader.load()
            logging.info(f"📄 Đã tải {len(documents)} trang từ tài liệu.")

            # --- Chia nhỏ ---
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(documents)
            logging.info(f"✂️ Đã chia thành {len(chunks)} đoạn.")

            # --- Lưu vào Chroma persistent ---
            persist_dir = os.path.join(CHROMA_DIR, self.session_id)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            self.vectorstore.persist()

            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            # --- Prompt ---
            template = """Sử dụng những thông tin được cung cấp dưới đây để trả lời câu hỏi một cách chính xác.
Nếu không có thông tin, hãy trả lời: 'Tôi không tìm thấy thông tin này trong tài liệu.' 
Không bịa đặt.

Thông tin: {context}

Câu hỏi: {question}

Trả lời chi tiết bằng tiếng Việt:"""
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            logging.info(f"✅ Tạo QA chain thành công cho session {self.session_id}")
            return True
        except Exception as e:
            logging.error(f"❌ Lỗi khi xử lý tài liệu: {e}")
            return False

    def chat(self, question):
        self.last_access = time.time()
        if not self.qa_chain:
            return {"error": "QA chain chưa được khởi tạo."}

        logging.info(f"🤖 [{self.session_id}] Q: {question}")
        result = self.qa_chain.invoke({"query": question})

        sources = []
        if 'source_documents' in result:
            sources = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result['source_documents']
            ]

        return {
            "answer": result.get("result", "Không có câu trả lời."),
            "sources": sources
        }

# =========================
# 🧹 DỌN SESSION CŨ
# =========================
def cleanup_sessions():
    while True:
        now = time.time()
        expired_sessions = [
            sid for sid, bot in list(SESSIONS.items())
            if now - bot.last_access > SESSION_TIMEOUT
        ]
        for sid in expired_sessions:
            logging.info(f"🧹 Dọn session {sid} do hết hạn.")
            # Xóa vectorstore trên ổ đĩa
            persist_dir = os.path.join(CHROMA_DIR, sid)
            if os.path.exists(persist_dir):
                for root, dirs, files in os.walk(persist_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(persist_dir)
            SESSIONS.pop(sid, None)
        time.sleep(60)

Thread(target=cleanup_sessions, daemon=True).start()

# =========================
# 🌐 API ENDPOINTS
# =========================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if not EMBEDDINGS_MODEL or not LLM:
        return jsonify({"success": False, "message": "Model chưa sẵn sàng"}), 503

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"success": False, "message": "Thiếu file"}), 400

    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        file_path = tmp.name

    try:
        session_id = str(uuid.uuid4())
        chatbot = RAGChatbot(embeddings_model=EMBEDDINGS_MODEL, llm=LLM, session_id=session_id)
        if chatbot.load_and_index(file_path):
            SESSIONS[session_id] = chatbot
            return jsonify({"success": True, "session_id": session_id, "message": "Tài liệu đã được xử lý."})
        else:
            return jsonify({"success": False, "message": "Xử lý tài liệu thất bại"}), 500
    except Exception as e:
        logging.error(f"❌ Lỗi upload: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        os.remove(file_path)

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.json
    session_id = data.get('session_id')
    question = data.get('question', '').strip()

    if not session_id or session_id not in SESSIONS:
        return jsonify({"success": False, "message": "Session không hợp lệ"}), 404
    if not question:
        return jsonify({"success": False, "message": "Câu hỏi trống"}), 400

    chatbot = SESSIONS[session_id]
    try:
        result = chatbot.chat(question)
        if "error" in result:
            return jsonify({"success": False, "message": result["error"]}), 400
        return jsonify({"success": True, **result})
    except Exception as e:
        logging.error(f"❌ Lỗi khi chat session {session_id}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# =========================
# 🚀 CHẠY ỨNG DỤNG
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
