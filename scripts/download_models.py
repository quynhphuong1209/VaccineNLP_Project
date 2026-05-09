import os
from huggingface_hub import snapshot_download

# Danh sách các repo cần tải
repos = [
    "quynhphuong1209/xlmr-multitask",
    "quynhphuong1209/phobert-multitask",
    "quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai"
]

base_dir = "huggingface"

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for repo in repos:
    folder_name = repo.split("/")[-1]
    target_dir = os.path.join(base_dir, folder_name)
    print(f"\n--- Đang tải {repo} vào {target_dir} ---")
    
    # Xóa thư mục nếu nó đang bị lỗi/trống để tải lại từ đầu cho chắc chắn
    if os.path.exists(target_dir) and not os.listdir(target_dir):
        import shutil
        shutil.rmtree(target_dir)

    try:
        snapshot_download(
            repo_id=repo,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            max_workers=1, # Tải tuần tự để tránh nghẽn/timeout trên Windows
            resume_download=True # Hỗ trợ tải tiếp nếu bị ngắt quãng
        )
        print(f"--- Hoàn thành {repo} ---")
    except Exception as e:
        print(f"Lỗi khi tải {repo}: {e}")

print("\n--- Đã tải xong tất cả các mô hình! ---")
