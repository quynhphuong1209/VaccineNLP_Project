# HƯỚNG PHÁT TRIỂN TƯƠNG LAI: REAL-TIME XAI VỚI LM STUDIO

Tài liệu này lưu trữ bản thiết kế kỹ thuật (Blueprint) chi tiết cho việc nâng cấp hệ thống Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam chạy Explainable AI (XAI) theo thời gian thực trên môi trường cục bộ (Localhost). Được phê duyệt và đóng băng vào ngày 23/04/2026 để ưu tiên hoàn thiện Luận văn. Sẵn sàng thực thi sau khi hoàn tất bảo vệ.1. Động lực (Motivation)Hệ thống hiện tại sử dụng cơ chế XAI Cache (lưu sẵn lý luận của 186 mẫu) để đảm bảo an toàn tuyệt đối khi Demo. Tuy nhiên, để chứng minh trọn vẹn năng lực "Zero-Cost Local Deployment" đối với các văn bản hoàn toàn mới (Out-of-distribution), hệ thống cần một động cơ suy luận LLM chạy ngầm (Inference Backend).2. Kiến trúc Đề xuất (Proposed Architecture)Tích hợp LM Studio làm máy chủ API cục bộ:Backend: LM Studio nạp mô hình Gemma-4 4B (định dạng GGUF) và mở cổng http://localhost:1234/v1 (Chuẩn OpenAI API).Frontend (Streamlit): Bổ sung nút gạt (Toggle) để chuyển đổi giữa 2 chế độ:Safe Mode: Gọi XAI từ Cache (Mặc định).Live Mode: Gửi văn bản mới xuống Localhost, gọi Gemma sinh lý luận thời gian thực (Streaming).3. Bản đồ Kỹ thuật (Implementation Roadmap)Bước 1: Xuất Mô hình sang GGUF (Kaggle/Colab)Cần thực hiện gộp trọng số (Merge Weights) trước khi xuất để tránh mất dữ liệu Fine-tuning:# Bước 1a: Merge LoRA vào base model trước khi export (BẮT BUỘC)
model = model.merge_and_unload()  

# Bước 1b: Export sang GGUF (Q4_K_M tối ưu cho RAM < 4GB)
model.save_pretrained_gguf(
    "gemma-4-vaccinenlp-gguf",
    tokenizer,
    quantization_method="q4_k_m"
)

# Bước 1c: Kiểm tra dung lượng file (Q4_K_M của 4B model khoảng 2.5-3GB)
import os
gguf_files = [f for f in os.listdir("gemma-4-vaccinenlp-gguf") if f.endswith(".gguf")]
for f in gguf_files:
    size_gb = os.path.getsize(f"gemma-4-vaccinenlp-gguf/{f}") / 1e9
    print(f"{f}: {size_gb:.2f} GB")
Lưu ý kỹ thuật: Unsloth hiện tại có thể export GGUF với metadata general.architecture = gemma3 do llama.cpp chưa hỗ trợ native hoàn toàn cho Gemma-4. Khi load vào LM Studio, cần thủ công chọn đúng Model Type trong UI nếu phần mềm không tự nhận diện.Bước 2: Cấu hình LM Studio (Chống rủi ro Template & OOM)Chat Template Override: Ghi đè Custom Template:User Prefix: <|turn>user\nAssistant Prefix: <|turn>model\nSystem Prompt: BẮT BUỘC chép chính xác System Instruction đã dùng khi huấn luyện để đảm bảo Output đúng định dạng parse:You are a highly rigorous Public Health Data Analyst in Vietnam. 
Your task is to analyze social media comments regarding vaccines and annotate them strictly.

CRITICAL INSTRUCTION: Do not be distracted by philosophical arguments, 
political rhetoric, or manipulative half-truths. Focus entirely on identifying 
the author's TRUE attitude toward VACCINATION.

Analyze the comment, reason step-by-step, and classify it based on these 3 criteria:
1. Misinformation: [Chính xác] or [Tin giả / Sai lệch] or [Không chắc chắn / Không liên quan]
2. Stance: [Ủng hộ] or [Phản đối] or [Trung lập]
3. Sentiment: [Tích cực] or [Tiêu cực] or [Trung tính]

The VERY LAST LINE of your response MUST exactly match this format:
Kết quả: <Misinformation Label> | <Stance Label> | <Sentiment Label>
Resource Capping: Bật GPU Offload, giới hạn Context Length = 1024 tokens.Bước 3: Mã nguồn Tích hợp StreamlitBổ sung đoạn code sau vào app/streamlit_demo.py để quản lý luồng Live XAI và Fallback:import openai
import time
import streamlit as st

def check_lmstudio_available(base_url="http://localhost:1234/v1", timeout=2):
    """Kiểm tra LM Studio có đang chạy không."""
    try:
        client = openai.OpenAI(base_url=base_url, api_key="lm-studio")
        client.models.list()
        return True
    except Exception:
        return False

def get_live_xai_reasoning(text: str, base_url="http://localhost:1234/v1") -> str:
    """
    Gọi Gemma-4 qua LM Studio API, streaming output.
    Trả về reasoning string, hoặc None nếu thất bại.
    """
    client = openai.OpenAI(base_url=base_url, api_key="lm-studio")
    prompt = f"Văn bản: {text}"
    
    try:
        stream = client.chat.completions.create(
            model="gemma-4-vaccinenlp",  # Cần khớp với ID trong LM Studio
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
            stream=True,
            timeout=30
        )
        
        full_response = ""
        placeholder = st.empty()
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                # Streaming hiển thị từng chữ ra màn hình
                placeholder.markdown(f"*{full_response}▌*")
        
        placeholder.empty()
        return full_response
        
    except openai.APIConnectionError:
        return None  # LM Studio không chạy → fallback về cache
    except Exception:
        return None

def render_xai_section(user_text: str, xai_cache: dict):
    """Render XAI section với toggle Safe/Live mode."""
    # Kiểm tra LM Studio availability một lần khi app load
    lmstudio_available = check_lmstudio_available()
    
    if lmstudio_available:
        mode = st.radio(
            "Chế độ XAI:",
            options=["🔒 Safe Mode (Cache)", "⚡ Live Mode (Gemma-4 Local)"],
            index=0,
            horizontal=True
        )
    else:
        mode = "🔒 Safe Mode (Cache)"
        st.caption("💡 LM Studio chưa chạy — đang dùng Safe Mode")
    
    if "Live Mode" in mode:
        st.markdown("**🧠 Gemma-4 đang phân tích...**")
        reasoning = get_live_xai_reasoning(user_text)
        
        if reasoning is None:
            st.warning("⚠️ Kết nối LM Studio thất bại. Chuyển về Safe Mode.")
            reasoning = xai_cache.get(user_text.strip())
    else:
        reasoning = xai_cache.get(user_text.strip())
    
    return reasoning
4. Ghi nhận Rủi ro cho Tương lai (Future Risks)Khi triển khai blueprint này, Team cần lưu ý giải quyết 2 rủi ro sau:Rủi ro về Parse Logic: Code hiện tại chỉ hiển thị text reasoning. Nếu kích hoạt Live Mode, cần tích hợp thêm hàm parse (Regex) để trích xuất 3 nhãn ở dòng "Kết quả:" của Gemma và so sánh trực tiếp độ lệch (discrepancy) với kết quả của PhoBERT.Rủi ro về Model Name: Biến model="gemma-4-vaccinenlp" đang bị hardcode. Cần bổ sung logic dùng client.models.list() để tự động fetch tên model đang được load thực tế trong LM Studio.