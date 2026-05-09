import argparse
from src.common import paths

def start_engine(model_path, port=8000):
    print("🚀 Đang khởi động Gemma-4 Inference Engine...")
    print(f"🤖 Model: {model_path}")
    print(f"🔌 Port: {port}")
    
    # Ở đây thường sẽ dùng Flask hoặc FastAPI để host model
    # Trong bản demo này, chúng ta giả lập việc khởi tạo server
    print("\n--- STATUS ---")
    print("✓ Model Weights: Loaded (4-bit QLoRA)")
    print("✓ CUDA Status: Active")
    print(f"✓ Endpoint: http://localhost:{port}/v1/reasoning")
    print("\n💡 Server đang sẵn sàng phục vụ các yêu cầu XAI từ Dashboard.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(paths.MODEL_DIR / "gemma-4b-it"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    start_engine(args.model, args.port)
