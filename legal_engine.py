import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VietGuard_Engine")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

class LegalClauseParser:
    """
    Parser chuyên dụng cho văn bản hợp đồng tiếng Việt.
    Bóc tách theo cấu trúc: Điều 1, Điều 2, Khoản... thay vì cắt vụn theo số ký tự.
    """
    @staticmethod
    def parse(contract_text: str) -> List[Dict[str, str]]:
        pattern = r"(?im)^(?:\s*)(Điều\s+\d+[\.:\s\-–—][^\n]*)"
        matches = list(re.finditer(pattern, contract_text))
        
        if not matches:
            paragraphs = [p.strip() for p in contract_text.split("\n\n") if len(p.strip()) > 40]
            clauses = []
            for i, p in enumerate(paragraphs, 1):
                clauses.append({
                    "clause_id": f"Mục {i}",
                    "title": p.split("\n")[0][:60],
                    "content": p
                })
            return clauses

        clauses = []
        for i in range(len(matches)):
            start_pos = matches[i].start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(contract_text)
            
            raw_chunk = contract_text[start_pos:end_pos].strip()
            title_line = matches[i].group(1).strip()
            
            id_match = re.search(r"(?i)Điều\s+\d+", title_line)
            clause_id = id_match.group(0).capitalize() if id_match else f"Điều {i+1}"
            
            clauses.append({
                "clause_id": clause_id,
                "title": title_line,
                "content": raw_chunk
            })
            
        return clauses

class LegalKnowledgeBase:
    """
    Cơ sở tri thức luật Việt Nam (Bộ luật Lao động, Dân sự, Luật Nhà ở).
    Sử dụng Hybrid Search (ChromaDB + BM25) để truy vấn luật chính xác.
    """
    def __init__(self, kb_dir: str = "knowledge_base", persist_dir: str = "chroma_legal_kb"):
        self.kb_dir = kb_dir
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        self.retriever = None
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        docs = []
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir, exist_ok=True)
            
        for filename in os.listdir(self.kb_dir):
            if filename.endswith(".md") or filename.endswith(".txt"):
                filepath = os.path.join(self.kb_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                articles = re.split(r"(?m)^(?=##\s+)", content)
                for art in articles:
                    art = art.strip()
                    if art:
                        docs.append(Document(page_content=art, metadata={"source": filename}))
                        
        if not docs:
            logger.warning("Cơ sở tri thức luật đang trống!")
            return

        logger.info(f"Nạp {len(docs)} điều luật vào Knowledge Base.")
        
        # Tối ưu: Nếu vectorstore đã tồn tại trên đĩa, load lại thay vì embed lại
        if os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0:
            logger.info("Tải Vectorstore có sẵn từ đĩa...")
            vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Tạo mới Chroma vectorstore và index embeddings...")
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # BM25 Sparse
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 2
        
        # Hybrid
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        logger.info("Hybrid Legal Retriever đã sẵn sàng.")

    def search_law(self, query: str) -> List[Document]:
        if not self.retriever:
            return []
        return self.retriever.invoke(query)

class ContractAuditor:
    """
    Hệ thống phân tích rủi ro hợp đồng, chấm điểm ĐỎ / VÀNG / XANH
    và đề xuất câu chữ đàm phán sửa đổi (Redlining).
    """
    def __init__(self, kb: LegalKnowledgeBase):
        self.kb = kb
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )

    def audit_clause(self, clause_info: Dict[str, str], contract_type: str = "Hợp đồng chung") -> Dict[str, Any]:
        clause_text = clause_info["content"]
        
        # Tìm các điều luật liên quan bằng Hybrid Search
        legal_docs = self.kb.search_law(clause_text)
        legal_context = "\n\n---\n\n".join([d.page_content for d in legal_docs]) if legal_docs else "Không có điều luật vi phạm."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Bạn là VietGuard AI - Chuyên gia pháp lý cao cấp chuyên rà soát rủi ro hợp đồng tại Việt Nam.
Hãy phân tích khách quan điều khoản hợp đồng được cung cấp và đối chiếu với Căn cứ pháp luật tham chiếu.

QUY TẮC PHÂN LOẠI MỨC ĐỘ RỦI RO (RẤT QUAN TRỌNG):
1. "RED" (RỦI RO CAO / TRÁI LUẬT CẤM):
   - Chỉ chọn RED khi điều khoản VI PHẠM RÕ RÀNG điều cấm của pháp luật hoặc bị pháp luật tuyên bố vô hiệu.
   - Ví dụ RED: Giữ bản chính CCCD/bằng đại học gốc; Bắt nộp tiền cọc/ký quỹ đi làm; Thử việc quá 60 ngày cho nhân viên văn phòng; Lương thử việc dưới 85%; Phạt tiền trừ thẳng vào lương; Phạt tiền khi nghỉ việc trong thời gian thử việc; Chủ nhà tự ý mở khóa vào phòng người thuê.
   - Điểm số: 4 hoặc 5.

2. "YELLOW" (RỦI RO TRUNG BÌNH / BẤT LỢI ĐƠN PHƯƠNG):
   - Điều khoản KHÔNG vi phạm luật cấm, nhưng cực kỳ bất bình đẳng hoặc cài bẫy gây thiệt thòi cho người ký.
   - Ví dụ YELLOW: Mất 100% tiền cọc trong mọi trường hợp chấm dứt trước hạn; Tự ý ấn định chi phí đào tạo nội bộ quá cao; Giá điện thu cố định cao hơn bình thường; Bắt người thuê chịu mọi hư hỏng kết cấu nhà ngầm.
   - Điểm số: 2 hoặc 3.

3. "GREEN" (AN TOÀN / HỢP PHÁP / THÔNG LỆ CHUẨN):
   - Điều khoản hoàn toàn bình thường, hợp lệ theo quy định pháp luật và thông lệ công bằng.
   - Ví dụ GREEN: Thanh toán tiền thuê đúng ngày mùng 5 hàng tháng; Nghĩa vụ bảo quản tài sản chung; Giờ làm việc 8 tiếng/ngày theo luật; Trả lương qua tài khoản ngân hàng.
   - Nếu điều khoản không có gì trái luật và không ép uổng bất thường, BẮT BUỘC chọn GREEN.
   - Điểm số: 1.

QUY TẮC TRẢ VỀ: Trả về DUY NHẤT một chuỗi JSON hợp lệ, không có bất kỳ văn bản nào ngoài JSON:
{{
  "risk_level": "RED" | "YELLOW" | "GREEN",
  "risk_score": 1-5,
  "summary": "Tóm tắt đánh giá ngắn gọn trong 1 câu",
  "legal_violation": "Tên điều luật nếu có vi phạm/liên quan (ví dụ: Điều 17 Bộ luật Lao động 2019) hoặc null nếu an toàn",
  "explanation": "Giải thích chi tiết vì sao điều khoản này nguy hiểm/bất lợi, hoặc vì sao an toàn",
  "redline_recommendation": "Đoạn văn bản mẫu đề xuất sửa đổi để người dùng đem đi thương lượng (nếu RED/YELLOW) hoặc null (nếu GREEN)"
}}"""),
            ("human", """LOẠI HỢP ĐỒNG: {contract_type}

ĐIỀU KHOẢN CẦN RÀ SOÁT:
\"\"\"{clause_text}\"\"\"

CĂN CỨ PHÁP LUẬT THAM CHIẾU:
\"\"\"{legal_context}\"\"\"""")
        ])

        chain = prompt_template | self.llm | StrOutputParser()
        raw_output = chain.invoke({
            "contract_type": contract_type,
            "clause_text": clause_text,
            "legal_context": legal_context
        })

        try:
            clean_json = raw_output.strip()
            json_match = re.search(r"\{.*\}", clean_json, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
            result = json.loads(clean_json)
        except Exception as e:
            logger.error(f"Lỗi parse JSON: {e}. Raw: {raw_output}")
            result = {
                "risk_level": "YELLOW",
                "risk_score": 3,
                "summary": "Cần kiểm tra kỹ lưỡng điều khoản này",
                "legal_violation": "Tham chiếu quy định pháp luật hiện hành",
                "explanation": raw_output[:300],
                "redline_recommendation": "Đề nghị xem xét điều chỉnh theo thỏa thuận công bằng giữa hai bên."
            }

        result["clause_id"] = clause_info["clause_id"]
        result["title"] = clause_info["title"]
        result["original_content"] = clause_text
        return result

    def audit_contract(self, contract_text: str, contract_type: str = "Tự động phát hiện") -> Dict[str, Any]:
        clauses = LegalClauseParser.parse(contract_text)
        logger.info(f"Đã bóc tách được {len(clauses)} điều khoản từ hợp đồng.")
        
        audited_clauses = []
        red_count = 0
        yellow_count = 0
        green_count = 0
        
        for c in clauses:
            res = self.audit_clause(c, contract_type)
            audited_clauses.append(res)
            lvl = res.get("risk_level", "GREEN")
            if lvl == "RED":
                red_count += 1
            elif lvl == "YELLOW":
                yellow_count += 1
            else:
                green_count += 1

        overall_status = "DANGEROUS" if red_count > 0 else ("WARNING" if yellow_count > 0 else "SAFE")
        
        return {
            "overall_status": overall_status,
            "summary_stats": {
                "total_clauses": len(clauses),
                "red_flags": red_count,
                "yellow_flags": yellow_count,
                "green_flags": green_count
            },
            "clauses": audited_clauses
        }
