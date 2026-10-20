import os
import io
import sys
import json
import logging
from typing import Dict, Any

# Fix encoding cho terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from legal_engine import LegalKnowledgeBase, ContractAuditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VietGuard_Benchmark")

def run_benchmark():
    benchmark_file = "legal_trap_benchmark.json"
    if not os.path.exists(benchmark_file):
        print(f"Lỗi: Không tìm thấy file {benchmark_file}")
        return

    with open(benchmark_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print("================================================================")
    print("   VIETGUARD AI - ĐÁNH GIÁ NĂNG LỰC PHÁT HIỆN BẪY HỢP ĐỒNG")
    print(f"   Tổng số test case thực tế: {len(test_cases)}")
    print("================================================================\n")

    print("1. Khởi tạo Legal Knowledge Base & Hybrid Retriever...")
    kb = LegalKnowledgeBase(kb_dir="knowledge_base", persist_dir="chroma_legal_kb")
    auditor = ContractAuditor(kb)

    results = []
    correct_risk_count = 0
    severe_trap_detected = 0
    severe_trap_total = 0

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- [Case {i}/{len(test_cases)}] {tc['category']} | {tc['trap_type']} ---")
        print(f"Nội dung: {tc['clause_text']}")
        print(f"Kỳ vọng: {tc['expected_risk']} | Luật liên quan: {tc['target_law']}")
        
        clause_obj = {
            "clause_id": f"Case_{tc['id']}",
            "title": tc["trap_type"],
            "content": tc["clause_text"]
        }
        
        audit_res = auditor.audit_clause(clause_obj, contract_type=tc["category"])
        pred_risk = audit_res.get("risk_level", "GREEN")
        
        # Đánh giá
        is_risk_match = (pred_risk == tc["expected_risk"])
        if is_risk_match:
            correct_risk_count += 1
            
        if tc["expected_risk"] == "RED":
            severe_trap_total += 1
            if pred_risk == "RED":
                severe_trap_detected += 1

        print(f" -> Dự đoán AI: {pred_risk} (Score: {audit_res.get('risk_score')}/5)")
        print(f" -> Căn cứ trích dẫn: {audit_res.get('legal_violation')}")
        print(f" -> Tóm tắt: {audit_res.get('summary')}")
        print(f" -> Gợi ý đàm phán (Redline): {audit_res.get('redline_recommendation')}")
        print(f" -> Kết quả so khớp: {'✅ CHÍNH XÁC' if is_risk_match else '⚠️ CHÊNH LỆCH'}")

        results.append({
            "test_case_id": tc["id"],
            "trap_type": tc["trap_type"],
            "expected_risk": tc["expected_risk"],
            "predicted_risk": pred_risk,
            "risk_score": audit_res.get("risk_score"),
            "legal_violation": audit_res.get("legal_violation"),
            "summary": audit_res.get("summary"),
            "redline": audit_res.get("redline_recommendation"),
            "is_correct": is_risk_match
        })

    accuracy = (correct_risk_count / len(test_cases)) * 100
    severe_recall = (severe_trap_detected / severe_trap_total * 100) if severe_trap_total > 0 else 100

    summary_report = {
        "total_test_cases": len(test_cases),
        "accuracy_rate": f"{accuracy:.1f}%",
        "severe_trap_recall": f"{severe_recall:.1f}% ({severe_trap_detected}/{severe_trap_total})",
        "details": results
    }

    with open("legal_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    print("\n================================================================")
    print("                    TỔNG KẾT BÁO CÁO BENCHMARK")
    print("================================================================")
    print(f"Tỷ lệ chính xác phân loại rủi ro: {accuracy:.1f}% ({correct_risk_count}/{len(test_cases)})")
    print(f"Độ nhạy bắt bẫy nghiêm trọng (RED Recall): {severe_recall:.1f}% ({severe_trap_detected}/{severe_trap_total})")
    print(f"Đã lưu chi tiết kết quả vào file: legal_benchmark_results.json")
    print("================================================================")

if __name__ == "__main__":
    run_benchmark()
