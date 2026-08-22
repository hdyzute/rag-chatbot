import os
import uuid
import time
import tempfile
import logging
from threading import Thread
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from duckduckgo_search import DDGS

# =========================
# 🔧 CẤU HÌNH & KHỞI TẠO
# =========================
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

# Cấu hình bảo mật file upload
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"success": False, "message": "Kích thước file quá lớn (Tối đa 10MB)"}), 413

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
    LLM = OllamaLLM(model="qwen2.5:3b", temperature=0.3, base_url=OLLAMA_BASE_URL)
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
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.session_id = session_id
        self.last_access = time.time()
        self.history = []  # Lưu lịch sử chat (tối đa 5 lượt)
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
            
            # --- Tạo Hybrid Retriever (Vector + Keyword) ---
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            self.bm25_retriever.k = 3
            
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
            
            logging.info(f"✅ Tạo Hybrid Vectorstore thành công cho session {self.session_id}")
            return True
        except Exception as e:
            logging.error(f"❌ Lỗi khi xử lý tài liệu: {e}")
            return False

    def chat(self, question):
        self.last_access = time.time()
        if not self.ensemble_retriever:
            return {"error": "Vectorstore chưa được khởi tạo."}

        logging.info(f"🤖 [{self.session_id}] Q: {question}")
        
        # 1. Phân loại câu hỏi sử dụng Router LLM Chain
        decision = "DIRECT"
        try:
            history_str = ""
            if self.history:
                history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.history])

            router_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", """Bạn là một AI Router Agent thông minh. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và chọn ra phương thức xử lý phù hợp nhất.
Trả về chính xác một trong ba từ khóa sau dưới dạng viết hoa (không thêm bất kỳ từ nào khác, không viết hoa thường, không giải thích, không chứa dấu câu):
- RETRIEVE: Nếu câu hỏi liên quan trực tiếp đến nội dung, thông tin hoặc chi tiết nằm trong tài liệu đã tải lên.
- WEB_SEARCH: Nếu câu hỏi hỏi về các thông tin thực tế, thời sự, thời tiết, tin tức sự kiện mới, nhân vật nổi tiếng hoặc kiến thức rộng lớn ngoài tài liệu.
- DIRECT: Nếu câu hỏi là chào hỏi, cảm ơn, hỏi thăm xã giao, hoặc hỏi về bản thân bạn (AI).

Lịch sử trò chuyện:
{chat_history}"""),
                    ("human", "{question}"),
                    ("system", "Phân loại (chỉ trả về một từ duy nhất: RETRIEVE, WEB_SEARCH hoặc DIRECT):")
                ])
                | self.llm
                | StrOutputParser()
            )
            raw_decision = router_chain.invoke({
                "chat_history": history_str,
                "question": question
            })
            decision = raw_decision.strip().upper()
            logging.info(f"🧠 Agent decision: {decision}")
        except Exception as e:
            logging.error(f"❌ Lỗi phân loại câu hỏi: {e}")
            decision = "DIRECT"

        # 2. Xử lý câu hỏi dựa trên quyết định phân loại
        sources = []
        answer = ""
        action = "DIRECT"

        if "RETRIEVE" in decision:
            action = "RETRIEVE"
            logging.info("🎯 Agent quyết định: RETRIEVE (Truy xuất tài liệu)")
            
            # Tái cấu trúc câu hỏi độc lập (rephrase) nếu có lịch sử
            standalone_question = question
            if self.history:
                try:
                    history_msgs = []
                    for q, a in self.history:
                        history_msgs.append(HumanMessage(content=q))
                        history_msgs.append(AIMessage(content=a))

                    rephrase_chain = (
                        ChatPromptTemplate.from_messages([
                            ("system", "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất, hãy tạo một câu hỏi độc lập (standalone question) bằng tiếng Việt để truy xuất thông tin từ tài liệu. Chỉ trả về câu hỏi độc lập đó, không thêm bất kỳ văn bản nào khác."),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{question}")
                        ])
                        | self.llm
                        | StrOutputParser()
                    )
                    standalone_question = rephrase_chain.invoke({
                        "chat_history": history_msgs,
                        "question": question
                    }).strip()
                    logging.info(f"🔄 Standalone retrieval query: {standalone_question}")
                except Exception as e:
                    logging.error(f"❌ Lỗi rephrase câu hỏi: {e}")
                    standalone_question = question

            # Truy xuất Hybrid DB
            try:
                docs = self.ensemble_retriever.invoke(standalone_question)
                # Ensemble Retriever có thể trả về nhiều hơn k, giới hạn lại top 4
                docs = docs[:4]
            except Exception as e:
                logging.error(f"❌ Lỗi truy xuất Hybrid Search: {e}")
                docs = []

            context = "\n\n".join([doc.page_content for doc in docs])

            # Trả lời dựa trên ngữ cảnh tài liệu
            try:
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", """Sử dụng những thông tin tài liệu được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác.
Nếu không có thông tin tương ứng trong tài liệu, hãy trả lời: 'Tôi không tìm thấy thông tin này trong tài liệu.' và dừng lại. Tuyệt đối không tự bịa đặt thông tin.

Thông tin tài liệu:
{context}"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                history_msgs = []
                for q, a in self.history:
                    history_msgs.append(HumanMessage(content=q))
                    history_msgs.append(AIMessage(content=a))

                qa_chain = qa_prompt | self.llm | StrOutputParser()
                answer = qa_chain.invoke({
                    "context": context,
                    "chat_history": history_msgs,
                    "question": question
                })
            except Exception as e:
                logging.error(f"❌ Lỗi trả lời RAG: {e}")
                answer = f"Không thể xử lý câu trả lời từ tài liệu: {e}"

            sources = [
                {"type": "document", "content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ]

        elif "WEB_SEARCH" in decision:
            action = "WEB_SEARCH"
            logging.info("🎯 Agent quyết định: WEB_SEARCH (Tìm kiếm Internet)")
            
            # Tạo từ khóa tìm kiếm tối ưu dựa trên câu hỏi và lịch sử
            search_query = question
            if self.history:
                try:
                    history_msgs = []
                    for q, a in self.history:
                        history_msgs.append(HumanMessage(content=q))
                        history_msgs.append(AIMessage(content=a))

                    query_chain = (
                        ChatPromptTemplate.from_messages([
                            ("system", "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất, hãy tạo một câu truy vấn ngắn gọn bằng tiếng Việt để tìm kiếm thông tin mới nhất trên công cụ tìm kiếm. Chỉ trả về từ khóa tìm kiếm đó, không thêm bất kỳ văn bản nào khác."),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{question}")
                        ])
                        | self.llm
                        | StrOutputParser()
                    )
                    search_query = query_chain.invoke({
                        "chat_history": history_msgs,
                        "question": question
                    }).strip()
                    logging.info(f"🔄 Web Search Query: {search_query}")
                except Exception as e:
                    logging.error(f"❌ Lỗi sinh từ khóa tìm kiếm: {e}")
                    search_query = question

            # Thực hiện tìm kiếm trên DuckDuckGo
            search_results = []
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(search_query, max_results=3))
                    search_results = results
            except Exception as e:
                logging.error(f"❌ Lỗi tìm kiếm DuckDuckGo: {e}")

            # Thiết lập ngữ cảnh từ kết quả tìm kiếm web
            context_list = []
            for r in search_results:
                context_list.append(f"Tiêu đề: {r.get('title')}\nĐường dẫn: {r.get('href')}\nTóm tắt: {r.get('body')}")
            context = "\n\n---\n\n".join(context_list)

            # Trả lời dựa trên thông tin tìm kiếm
            try:
                web_prompt = ChatPromptTemplate.from_messages([
                    ("system", """Bạn là một trợ lý AI thông minh. Sử dụng kết quả tìm kiếm Internet dưới đây để trả lời câu hỏi của người dùng một cách chính xác và khách quan nhất.
Hãy giải thích thông tin rõ ràng và dẫn nguồn tham khảo nếu cần.

Kết quả tìm kiếm Internet:
{context}"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                history_msgs = []
                for q, a in self.history:
                    history_msgs.append(HumanMessage(content=q))
                    history_msgs.append(AIMessage(content=a))

                web_chain = web_prompt | self.llm | StrOutputParser()
                answer = web_chain.invoke({
                    "context": context,
                    "chat_history": history_msgs,
                    "question": question
                })
            except Exception as e:
                logging.error(f"❌ Lỗi trả lời từ Web: {e}")
                answer = f"Không thể xử lý câu trả lời từ kết quả tìm kiếm: {e}"

            sources = [
                {"type": "web", "title": r.get("title"), "url": r.get("href"), "content": r.get("body")}
                for r in search_results
            ]

        else:
            action = "DIRECT"
            logging.info("🎯 Agent quyết định: DIRECT (Trò chuyện trực tiếp)")
            
            # Trả lời trực tiếp qua kiến thức chung của mô hình
            try:
                direct_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Bạn là một trợ lý AI thông minh và hữu ích. Hãy trò chuyện với người dùng một cách thân thiện, lịch sự và tự nhiên nhất."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                history_msgs = []
                for q, a in self.history:
                    history_msgs.append(HumanMessage(content=q))
                    history_msgs.append(AIMessage(content=a))

                direct_chain = direct_prompt | self.llm | StrOutputParser()
                answer = direct_chain.invoke({
                    "chat_history": history_msgs,
                    "question": question
                })
            except Exception as e:
                logging.error(f"❌ Lỗi trò chuyện trực tiếp: {e}")
                answer = f"Không thể xử lý câu trả lời trực tiếp: {e}"

        # 3. Lưu lại lịch sử hội thoại
        self.history.append((question, answer))
        if len(self.history) > 5:
            self.history.pop(0)

        return {
            "action": action,
            "answer": answer,
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
        return jsonify({"success": False, "message": "Model chưa sẵn sàng. Hãy chạy lệnh 'ollama pull qwen2.5:3b' và 'ollama pull nomic-embed-text' trước."}), 503

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"success": False, "message": "Thiếu file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Định dạng file không được hỗ trợ (Chỉ nhận PDF và TXT)"}), 400

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
