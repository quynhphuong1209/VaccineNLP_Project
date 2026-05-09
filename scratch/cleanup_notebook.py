import json
import sys
import os

def cleanup_notebook(notebook_path):
    """Xóa các output cell để giảm dung lượng file notebook"""
    if not os.path.exists(notebook_path):
        print(f"❌ Không tìm thấy file: {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"🧹 Đã dọn dẹp sạch sẽ: {notebook_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cleanup_notebook(sys.argv[1])
    else:
        print("💡 Sử dụng: python scratch/cleanup_notebook.py <đường_dẫn_file.ipynb>")
