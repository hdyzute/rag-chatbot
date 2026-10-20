import os
import uuid
import time
import json
import logging
import tempfile
from typing import Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pypdf import PdfReader

from legal_engine import LegalClauseParser, LegalKnowledgeBase, ContractAuditor
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VietGuard_App")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15MB

# Khởi tạo Knowledge Base & Auditor
logger.info("Khởi động VietGuard AI Engine...")
LEGAL_KB = LegalKnowledgeBase(kb_dir="knowledge_base", persist_dir="chroma_legal_kb")
AUDITOR = ContractAuditor(LEGAL_KB)
CHAT_LLM = OllamaLLM(model="qwen2.5:3b", base_url="http://localhost:11434", temperature=0.2)

# Quản lý Session trên RAM (kèm thời gian hết hạn)
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRY = 3600  # 1 giờ

def cleanup_old_sessions():
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s.get("last_active", 0) > SESSION_EXPIRY]
    for sid in expired:
        del SESSIONS[sid]

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/sample-contracts", methods=["GET"])
def get_sample_contracts():
    samples = [
        {
            "id": "sample_rental",
            "title": "Hợp đồng thuê phòng trọ (Bẫy cọc & tiền điện)",
            "category": "Thuê nhà",
            "filename": "hop_dong_thue_nha_bay_coc.txt",
            "description": "Chứa các bẫy thực tế: Tịch thu 100% cọc vô điều kiện, giá điện 4.500đ/kWh, chủ nhà tự ý vào phòng."
        },
        {
            "id": "sample_labor",
            "title": "Hợp đồng lao động & Cam kết đào tạo (Bẫy giữ bằng & phạt tiền)",
            "category": "Lao động",
            "filename": "hop_dong_lao_dong_bay_giay_to_phat_tien.txt",
            "description": "Chứa các bẫy trái luật: Giữ CCCD và bằng gốc, thử việc 90 ngày, phạt tiền đi trễ, phạt 60 triệu đào tạo."
        }
    ]
    return jsonify({"success": True, "samples": samples})

@app.route("/api/sample-contract/<contract_id>", methods=["GET"])
def get_sample_content(contract_id):
    filename_map = {
        "sample_rental": "hop_dong_thue_nha_bay_coc.txt",
        "sample_labor": "hop_dong_lao_dong_bay_giay_to_phat_tien.txt"
    }
    if contract_id not in filename_map:
        return jsonify({"success": False, "message": "Không tìm thấy hợp đồng mẫu."}), 404
        
    path = os.path.join("sample_contracts", filename_map[contract_id])
    if not os.path.exists(path):
        return jsonify({"success": False, "message": "File không tồn tại."}), 404
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return jsonify({"success": True, "contract_text": content})

@app.route("/api/audit", methods=["POST"])
def audit_contract_endpoint():
    cleanup_old_sessions()
    
    contract_text = ""
    contract_type = "Tự động phát hiện"
    
    # 1. Đọc từ file upload hoặc text gửi lên
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"success": False, "message": "File không hợp lệ."}), 400
            
        fname = uploaded_file.filename.lower()
        if fname.endswith(".pdf"):
            try:
                reader = PdfReader(uploaded_file.stream)
                text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                contract_text = "\n\n".join(text_pages)
            except Exception as e:
                return jsonify({"success": False, "message": f"Lỗi đọc file PDF: {str(e)}"}), 400
        elif fname.endswith(".txt") or fname.endswith(".md"):
            contract_text = uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            return jsonify({"success": False, "message": "Hệ thống hỗ trợ file .pdf hoặc .txt"}), 400
    else:
        data = request.get_json(silent=True) or {}
        contract_text = data.get("contract_text", "").strip()
        contract_type = data.get("contract_type", "Tự động phát hiện")

    if not contract_text or len(contract_text) < 50:
        return jsonify({"success": False, "message": "Nội dung hợp đồng quá ngắn hoặc không có chữ."}), 400

    # Phán đoán loại hợp đồng nếu để tự động
    lower_text = contract_text.lower()
    if "lao động" in lower_text or "lương" in lower_text or "thử việc" in lower_text:
        contract_type = "Hợp đồng lao động"
    elif "thuê nhà" in lower_text or "phòng trọ" in lower_text or "tiền phòng" in lower_text:
        contract_type = "Hợp đồng thuê nhà"

    logger.info(f"Bắt đầu audit hợp đồng ({contract_type})...")
    audit_result = AUDITOR.audit_contract(contract_text, contract_type=contract_type)
    
    # Khởi tạo session cho phép hỏi đáp về hợp đồng này
    session_id = str(uuid.uuid4())
    clauses = audit_result["clauses"]
    
    # Tạo vectorstore tạm trên RAM cho các điều khoản của hợp đồng
    clause_docs = [
        Document(
            page_content=f"{c.get('clause_id', '')} - {c.get('title', '')}:\n{c.get('original_content', '')}",
            metadata={"clause_id": c.get("clause_id", ""), "risk_level": c.get("risk_level", "GREEN")}
        )
        for c in clauses
    ]
    
    contract_vectorstore = Chroma.from_documents(
        documents=clause_docs,
        embedding=LEGAL_KB.embeddings
    )
    
    SESSIONS[session_id] = {
        "contract_type": contract_type,
        "raw_text": contract_text,
        "clauses": clauses,
        "vectorstore": contract_vectorstore,
        "last_active": time.time(),
        "chat_history": []
    }

    return jsonify({
        "success": True,
        "session_id": session_id,
        "contract_type": contract_type,
        "overall_status": audit_result["overall_status"],
        "summary_stats": audit_result["summary_stats"],
        "clauses": clauses
    })

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    cleanup_old_sessions()
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    question = data.get("message", "").strip()

    if not session_id or session_id not in SESSIONS:
        return jsonify({"success": False, "message": "Phiên làm việc không tồn tại hoặc đã hết hạn. Vui lòng tải lại hợp đồng."}), 400

    if not question:
        return jsonify({"success": False, "message": "Câu hỏi không được để trống."}), 400

    session_data = SESSIONS[session_id]
    session_data["last_active"] = time.time()
    contract_vs = session_data["vectorstore"]

    # 1. Tìm điều khoản hợp đồng liên quan
    contract_chunks = contract_vs.similarity_search(question, k=2)
    contract_context = "\n\n".join([d.page_content for d in contract_chunks])

    # 2. Tìm căn cứ pháp lý liên quan từ Legal Knowledge Base
    legal_docs = LEGAL_KB.search_law(question)
    legal_context = "\n\n".join([d.page_content for d in legal_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là VietGuard Chatbot - Trợ lý pháp lý thông minh bảo vệ quyền lợi người ký hợp đồng tại Việt Nam.
Hãy trả lời câu hỏi của người dùng một cách chính xác, thân thiện, dễ hiểu, bám sát các điều khoản trong hợp đồng của họ và đối chiếu với quy định pháp luật Việt Nam.

NỘI DUNG HỢP ĐỒNG CỦA NGƯỜI DÙNG:
\"\"\"{contract_context}\"\"\"

CĂN CỨ PHÁP LUẬT THAM CHIẾU:
\"\"\"{legal_context}\"\"\"

HƯỚNG DẪN TRẢ LỜI:
- Trả lời thẳng vào trọng tâm câu hỏi.
- Nếu điều khoản trong hợp đồng bất lợi hoặc trái luật, hãy chỉ rõ điều luật bảo vệ họ và khuyên họ cách xử lý khôn khéo.
- Giữ phong thái chuyên nghiệp, thấu cảm và tự tin."""),
        ("human", "{question}")
    ])

    chain = prompt | CHAT_LLM | StrOutputParser()
    answer = chain.invoke({
        "contract_context": contract_context,
        "legal_context": legal_context,
        "question": question
    })

    return jsonify({
        "success": True,
        "answer": answer,
        "cited_clauses": [d.metadata.get("clause_id") for d in contract_chunks]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
