import sys
import subprocess
import os
import time

# Load env variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

token = os.environ.get("NGROK_API_TOKEN", "").strip()
port = 1234

if not token:
    print("❌ ERROR: NGROK_API_TOKEN is not set in your .env file!")
    print("Vui lòng mở file .env và điền mã NGROK_API_TOKEN của bạn.")
    sys.exit(1)

print("🚀 Khởi động đường hầm Ngrok kết nối với LM Studio (Port 1234)...")

try:
    from pyngrok import ngrok, conf
    if token:
        ngrok.set_auth_token(token)
    
    # Open an HTTP tunnel on port 1234
    # Note: pyngrok will download ngrok binary automatically if not present!
    public_url = ngrok.connect(port, "http")
    
    print("\n" + "="*70)
    print("🎉 ĐƯỜNG HẦM NGROK ĐÃ ĐƯỢC THIẾT LẬP THÀNH CÔNG!")
    print(f"🔗 Public URL:  {public_url.public_url}")
    print(f"💡 API Endpoint: {public_url.public_url}/v1")
    print(f"🔒 Cổng local:   {port}")
    print("="*70 + "\n")
    print("👉 HƯỚNG DẪN ĐỒNG BỘ TRÊN HUGGING FACE SPACES (ONLINE):")
    print("1. Truy cập cài đặt (Settings) của HF Space của bạn.")
    print("2. Cập nhật các biến cấu hình (Secrets) sau để kết nối trực tiếp với GPU ở nhà:")
    print(f"   - LM_STUDIO_URL   = {public_url.public_url}/v1")
    print(f"   - LM_STUDIO_MODEL = gemma-4-e4b-vaccine-xai-merged")
    print("   - LM_API_TOKEN    = (Token bảo mật LM Studio của bạn)")
    print("\nNhấn Ctrl + C để tắt đường hầm.")
    
    # Keep the tunnel alive
    while True:
        time.sleep(1)

except ImportError:
    print("📦 Thư viện 'pyngrok' chưa được cài đặt. Đang tiến hành cài đặt tự động...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        print("✅ Đã cài đặt xong 'pyngrok'. Khởi động lại script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as install_err:
        print(f"❌ Không thể cài đặt pyngrok tự động: {install_err}")
        print("Đang chuyển sang chạy trực tiếp CLI của hệ điều hành...")
        os.system(f"ngrok config add-authtoken {token}")
        os.system(f"ngrok http {port}")
except Exception as e:
    print(f"❌ Lỗi khi khởi chạy pyngrok: {e}")
    print("🔄 Đang thử chạy trực tiếp bằng lệnh ngrok của Windows...")
    try:
        os.system(f"ngrok config add-authtoken {token}")
        os.system(f"ngrok http {port}")
    except Exception as cli_error:
        print(f"❌ Lỗi chạy trực tiếp CLI: {cli_error}")
        print("💡 Hãy đảm bảo bạn đã tải ngrok.exe và đưa vào PATH hệ thống.")
