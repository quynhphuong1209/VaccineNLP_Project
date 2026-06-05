import os
import sys
from huggingface_hub import HfApi

# Reconfigure stdout to support UTF-8 on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("VaccineNLP_TOKEN")
REPO_ID = "quynhphuong1209/VaccineNLP_demo"

def deploy():
    if not TOKEN:
        print("❌ Không tìm thấy biến môi trường HF_TOKEN hoặc VaccineNLP_TOKEN. Hãy đặt token Hugging Face trước khi chạy.")
        print("   Ví dụ: setx VaccineNLP_TOKEN \"hf_xxx...\"")
        return

    print(f"🚀 Khởi tạo Hugging Face API...")
    api = HfApi(token=TOKEN)
    
    print(f"📦 Đang tải lên toàn bộ file trong thư mục 'app_gradio' lên Space '{REPO_ID}'...")
    try:
        api.upload_folder(
            folder_path="app_gradio",
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="style(ui): premium dark navy + teal neon redesign v3.0 — glassmorphism, shimmer animations, glow effects"
        )
        print("\n✅ Cập nhật Space thành công! Bạn có thể kiểm tra tại: https://huggingface.co/spaces/quynhphuong1209/VaccineNLP_demo")
    except Exception as e:
        print(f"\n❌ Lỗi khi tải lên Space: {e}")

if __name__ == "__main__":
    deploy()
