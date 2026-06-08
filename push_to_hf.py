import os
import shutil
import subprocess
import sys
import stat

# Define directories
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_GRADIO_DIR = os.path.join(WORKSPACE_DIR, "app_gradio")
APP_DIR = os.path.join(WORKSPACE_DIR, "app")
TMP_CLONE_DIR = os.path.join(WORKSPACE_DIR, "hf_space_tmp")
HF_SPACE_URL = "https://huggingface.co/spaces/quynhphuong1209/VaccineNLP_Ban_chinh_thuc"

def run_cmd(args, cwd=None):
    print(f"Running: {' '.join(args)} (cwd: {cwd})")
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False, result.stderr
    print(result.stdout)
    return True, result.stdout

def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Failed to remove {path}: {e}")

def main():
    print("=== HuggingFace Space Sync Script ===")
    
    # 1. Check if git is available
    print("Checking git installation...")
    success, _ = run_cmd(["git", "--version"])
    if not success:
        print("Git is not found in your PATH. Please make sure Git is installed and added to PATH.")
        sys.exit(1)

    # 2. Synchronize data files locally into app_gradio/data
    print("Synchronizing local data files into app_gradio/data...")
    local_data_dir = os.path.join(APP_GRADIO_DIR, "data")
    os.makedirs(local_data_dir, exist_ok=True)

    src_cache = os.path.join(APP_DIR, "xai_cache.json")
    src_temp = os.path.join(APP_DIR, "temperature_params.json")
    src_bench = os.path.join(WORKSPACE_DIR, "experiments", "results", "benchmark_results.json")

    dst_cache = os.path.join(local_data_dir, "xai_cache.json")
    dst_temp = os.path.join(local_data_dir, "temperature_params.json")
    dst_bench = os.path.join(local_data_dir, "benchmark_results.json")

    if os.path.exists(src_cache):
        print(f"Copying {src_cache} -> {dst_cache}...")
        shutil.copy2(src_cache, dst_cache)
    else:
        print(f"Warning: {src_cache} not found!")

    if os.path.exists(src_temp):
        print(f"Copying {src_temp} -> {dst_temp}...")
        shutil.copy2(src_temp, dst_temp)
    else:
        print(f"Warning: {src_temp} not found!")

    if os.path.exists(src_bench):
        print(f"Copying {src_bench} -> {dst_bench}...")
        shutil.copy2(src_bench, dst_bench)
    else:
        print(f"Preserving existing benchmark_results.json in app_gradio/data (if any).")

    # 3. Clean up previous clone if exists
    if os.path.exists(TMP_CLONE_DIR):
        print(f"Cleaning up old temporary directory: {TMP_CLONE_DIR}...")
        try:
            shutil.rmtree(TMP_CLONE_DIR, onerror=remove_readonly)
        except Exception as e:
            print(f"Warning: Could not fully delete directory via script: {e}")
            print("Please delete the folder 'hf_space_tmp' manually if the script fails.")

    # 4. Clone the HF Space repo
    print(f"Cloning Hugging Face Space repository from: {HF_SPACE_URL}...")
    success, _ = run_cmd(["git", "clone", HF_SPACE_URL, "hf_space_tmp"], cwd=WORKSPACE_DIR)
    if not success:
        print("\nFailed to clone repo. If it requires authentication, you can try standard HTTPS format with token:")
        print("  git clone https://<username>:<token>@huggingface.co/spaces/quynhphuong1209/VaccineNLP_Ban_chinh_thuc hf_space_tmp")
        print("\nOr configure git credentials helper. Please run cloning manually if needed.")
        sys.exit(1)

    # 5. Clean up temporary clone directory EXCEPT .git and .gitattributes
    print("Cleaning up old files in Hugging Face Space clone...")
    for item in os.listdir(TMP_CLONE_DIR):
        if item in [".git", ".gitattributes"]:
            continue
        path = os.path.join(TMP_CLONE_DIR, item)
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=remove_readonly)
        else:
            try:
                os.chmod(path, stat.S_IWRITE)
                os.remove(path)
            except Exception as e:
                print(f"Warning: could not delete {path}: {e}")

    # 6. Copy app_gradio files into temporary clone
    print(f"Copying files from app_gradio to temporary clone...")
    for item in os.listdir(APP_GRADIO_DIR):
        s = os.path.join(APP_GRADIO_DIR, item)
        d = os.path.join(TMP_CLONE_DIR, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # 7. Clean up __pycache__ and .pyc files inside temporary clone before git add (double safety)
    print("Purging any temporary cache files...")
    for root, dirs, files in os.walk(TMP_CLONE_DIR):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), onerror=remove_readonly)
                dirs.remove(d)
        for f in files:
            if f.endswith(".pyc"):
                path = os.path.join(root, f)
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.remove(path)
                except Exception:
                    pass

    # 8. Git commit & push
    print("Staging and committing files to HF Space...")
    success, _ = run_cmd(["git", "add", "."], cwd=TMP_CLONE_DIR)
    if not success:
        sys.exit(1)

    # Check if there are any changes
    success, status_out = run_cmd(["git", "status", "--porcelain"], cwd=TMP_CLONE_DIR)
    if not success:
        sys.exit(1)
    
    if not status_out.strip():
        print("No changes detected. The Space is already up to date!")
        try:
            shutil.rmtree(TMP_CLONE_DIR, onerror=remove_readonly)
        except Exception:
            pass
        sys.exit(0)

    # Show changes for user visibility
    print("Changes staged for commit:")
    print(status_out)

    success, _ = run_cmd(["git", "commit", "-m", "Deploy app_gradio to Hugging Face Spaces (clean mirror)"], cwd=TMP_CLONE_DIR)
    if not success:
        sys.exit(1)

    print("Pushing to HuggingFace Spaces. You may be asked for your credentials (username & access token as password)...")
    success, _ = run_cmd(["git", "push", "origin", "main"], cwd=TMP_CLONE_DIR)
    if not success:
        print("\nPush failed. Please go to 'hf_space_tmp' directory and run 'git push origin main' manually to authenticate.")
        sys.exit(1)

    print("\nSuccessfully pushed app_gradio to Hugging Face Space!")
    
    # 9. Cleanup
    print("Cleaning up temporary directory...")
    try:
        shutil.rmtree(TMP_CLONE_DIR, onerror=remove_readonly)
    except Exception:
        pass
    print("Done!")

if __name__ == "__main__":
    main()
