import os

def generate_structure():
    base_path = r"c:\Users\dinhl\Downloads\VaccineNLP_ĐỒ_ÁN"
    output_file = os.path.join(base_path, "FOLDER_STRUCTURE.md")
    
    icons = {
        "configs": "⚙️", "datasets": "📊", "models": "🤖", "notebook": "📔",
        "output": "📈", "script": "🔧", "src": "💻", "app": "📱", "docs": "📚",
        ".kaggle": "📊", "README.md": "📝", "requirements.txt": "📦", ".env": "🔐"
    }

    descriptions = {
        "configs": "Thư mục chứa các file cấu hình JSON cho hệ thống",
        "datasets": "Thư mục quản lý dữ liệu theo kiến trúc Medallion (Raw -> Processed)",
        "models": "Lưu trữ trọng số và checkpoints của các mô hình (PhoBERT, XLM-R, Gemma)",
        "notebook": "Các bản thử nghiệm Jupyer Notebook cho training và EDA",
        "output": "Kết quả xuất ra từ các thực nghiệm và dữ liệu tải từ Kaggle",
        "script": "Các script tự động hóa thu thập và xử lý dữ liệu",
        "src": "Mã nguồn chính của dự án (pipeline, modeling, preprocessing)",
        ".kaggle": "Thông tin cấu hình API Kaggle"
    }

    content = "# 📁 Cây Thư Mục Dự Án VaccineNLP (Cập nhật tự động)\n\n"
    content += "## 📊 Tổng Quan Kiến Trúc Thực Tế\n\n```\nVaccineNLP_ĐỒ_ÁN/\n│\n"

    # Liệt kê các thư mục và file ở root
    items = sorted(os.listdir(base_path))
    
    for item in items:
        if item.startswith('.') and item != '.kaggle': continue
        full_path = os.path.join(base_path, item)
        icon = icons.get(item, "📂")
        
        if os.path.isdir(full_path):
            content += f"├── {icon} {item}/".ljust(40) + f"# {descriptions.get(item, 'Thư mục dự án')}\n"
            # Thêm một cấp con nếu là thư mục quan trọng
            try:
                sub_items = sorted(os.listdir(full_path))[:5] # Chỉ lấy 5 item đầu cho gọn
                for sub in sub_items:
                    content += f"│   ├── {sub}\n"
                if len(os.listdir(full_path)) > 5:
                    content += f"│   └── ...\n"
            except: pass
            content += "│\n"
        else:
            content += f"├── {icon} {item}\n"

    content += "```\n\n---\n*Tài liệu này được tạo tự động bởi Antigravity AI.*"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ Đã tạo thành công file: {output_file}")

if __name__ == "__main__":
    generate_structure()
