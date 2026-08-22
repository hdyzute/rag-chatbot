import os
import json
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)

# 1. Cấu hình Model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model="qwen2.5:3b", temperature=0.1, base_url=OLLAMA_BASE_URL)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)

# 2. Tập dữ liệu Ground Truth (FPT Annual Report 2023)
eval_dataset = [
    {
        "question": "Năm 2023, Tập đoàn FPT đạt doanh thu và lợi nhuận trước thuế là bao nhiêu?",
        "ground_truth": "Doanh thu hợp nhất đạt 52.618 tỷ đồng và lợi nhuận trước thuế đạt 9.203 tỷ đồng."
    },
    {
        "question": "Thị trường nước ngoài nào của khối công nghệ là điểm sáng trong năm 2023 và tăng trưởng bao nhiêu?",
        "ground_truth": "Thị trường Nhật Bản là điểm sáng với mức tăng trưởng 43,4%."
    },
    {
        "question": "Doanh thu của mảng Dịch vụ Công nghệ thông tin trong nước đạt bao nhiêu và lợi nhuận thay đổi thế nào?",
        "ground_truth": "Doanh thu 7.161 tỷ đồng (tăng 4,6%) và lợi nhuận trước thuế đạt 379 tỷ đồng (giảm 2,6%)."
    }
]

# 3. Chuẩn bị Indexing với Hybrid Search
def setup_hybrid_retriever(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

# 4. Chạy Pipeline RAG
def run_rag(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sử dụng thông tin sau để trả lời câu hỏi ngắn gọn:\n{context}"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, context

# 5. LLM-as-a-Judge Evaluation Prompts
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một giám khảo AI chuyên đánh giá hệ thống RAG. 
Bạn sẽ chấm điểm (từ 1 đến 5) cho 3 tiêu chí sau. Trả về đúng định dạng JSON.
1. Context Relevance: Ngữ cảnh được truy xuất có chứa thông tin để trả lời câu hỏi không?
2. Faithfulness: Câu trả lời có dựa hoàn toàn vào ngữ cảnh không (hay bịa đặt)?
3. Answer Relevance: Câu trả lời có giải quyết đúng câu hỏi so với đáp án gốc (Ground Truth) không?

Định dạng JSON yêu cầu:
{{
    "context_relevance": <điểm>,
    "faithfulness": <điểm>,
    "answer_relevance": <điểm>,
    "reasoning": "<Giải thích ngắn gọn>"
}}
"""),
    ("human", """Câu hỏi: {question}
Ngữ cảnh truy xuất: {context}
Đáp án gốc (Ground Truth): {ground_truth}
Câu trả lời của RAG: {answer}""")
])

eval_chain = eval_prompt | llm | StrOutputParser()

# 6. Thực thi Evaluation
if __name__ == "__main__":
    logging.info("Bắt đầu khởi tạo hệ thống Hybrid Retriever...")
    retriever = setup_hybrid_retriever("sample_data.txt")
    
    results = []
    
    for idx, item in enumerate(eval_dataset):
        logging.info(f"Đang đánh giá câu hỏi {idx+1}/{len(eval_dataset)}...")
        q = item["question"]
        gt = item["ground_truth"]
        
        # Chạy RAG
        ans, ctx = run_rag(q, retriever)
        
        # Chạy LLM Judge
        try:
            eval_res_str = eval_chain.invoke({
                "question": q,
                "context": ctx,
                "ground_truth": gt,
                "answer": ans
            })
            
            # Xử lý text để lấy JSON an toàn
            start = eval_res_str.find('{')
            end = eval_res_str.rfind('}') + 1
            if start != -1 and end != -1:
                eval_json = json.loads(eval_res_str[start:end])
            else:
                eval_json = {"error": "Invalid JSON format from judge."}
                
        except Exception as e:
            eval_json = {"error": str(e)}
            
        results.append({
            "question": q,
            "rag_answer": ans,
            "metrics": eval_json
        })
        
    # Ghi kết quả
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    logging.info("Hoàn tất đánh giá! Đã lưu kết quả vào evaluation_results.json")
    for r in results:
        print(f"\nQ: {r['question']}\nA: {r['rag_answer']}\nScores: {r['metrics']}")
