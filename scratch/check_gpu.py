import torch
import sys

def check_env():
    print("--- 🖥️ KIỂM TRA MÔI TRƯỜNG HỆ THỐNG ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ CÓ' if cuda_available else '❌ KHÔNG'}")
    
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ Cảnh báo: Bạn đang chạy trên CPU. Việc huấn luyện PhoBERT/Gemma sẽ rất chậm.")

if __name__ == "__main__":
    check_env()
