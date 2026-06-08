import os
import shutil
import subprocess
import sys

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

def main():
    print("=== HuggingFace Space Sync Script ===")
    
    # 1. Check if git is available
    print("Checking git installation...")
    success, _ = run_cmd(["git", "--version"])
    if not success:
        print("Git is not found in your PATH. Please make sure Git is installed and added to PATH.")
        sys.exit(1)

    # 2. Clean up previous clone if exists
    if os.path.exists(TMP_CLONE_DIR):
        print(f"Cleaning up old temporary directory: {TMP_CLONE_DIR}...")
        shutil.rmtree(TMP_CLONE_DIR, ignore_errors=True)

    # 3. Clone the HF Space repo
    print(f"Cloning Hugging Face Space repository from: {HF_SPACE_URL}...")
    success, _ = run_cmd(["git", "clone", HF_SPACE_URL, "hf_space_tmp"], cwd=WORKSPACE_DIR)
    if not success:
        print("\nFailed to clone repo. If it requires authentication, you can try standard HTTPS format with token:")
        print("  git clone https://<username>:<token>@huggingface.co/spaces/quynhphuong1209/VaccineNLP_Ban_chinh_thuc hf_space_tmp")
        print("\nOr configure git credentials helper. Please run cloning manually if needed.")
        sys.exit(1)

    # 4. Copy app_gradio files
    print(f"Copying files from app_gradio to temporary clone...")
    for item in os.listdir(APP_GRADIO_DIR):
        s = os.path.join(APP_GRADIO_DIR, item)
        d = os.path.join(TMP_CLONE_DIR, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # 5. Create data directory in clone
    clone_data_dir = os.path.join(TMP_CLONE_DIR, "data")
    os.makedirs(clone_data_dir, exist_ok=True)

    # 6. Copy cache and parameters files if they exist locally
    # Local paths in 'app/'
    src_cache = os.path.join(APP_DIR, "xai_cache.json")
    src_temp = os.path.join(APP_DIR, "temperature_params.json")
    
    dst_cache = os.path.join(clone_data_dir, "xai_cache.json")
    dst_temp = os.path.join(clone_data_dir, "temperature_params.json")

    if os.path.exists(src_cache):
        print(f"Copying {src_cache} -> {dst_cache}...")
        shutil.copy2(src_cache, dst_cache)
    else:
        print(f"Warning: {src_cache} not found! Cached XAI explanation will not be pre-loaded.")

    if os.path.exists(src_temp):
        print(f"Copying {src_temp} -> {dst_temp}...")
        shutil.copy2(src_temp, dst_temp)

    # 7. Check if benchmark_results.json is already in the clone, if not, check other places
    dst_benchmark = os.path.join(clone_data_dir, "benchmark_results.json")
    if os.path.exists(dst_benchmark):
        print(f"Preserving existing benchmark_results.json in the Space repository.")
    else:
        # Check if we have one in experiments/results/
        src_bench = os.path.join(WORKSPACE_DIR, "experiments", "results", "benchmark_results.json")
        if os.path.exists(src_bench):
            print(f"Copying {src_bench} -> {dst_benchmark}...")
            shutil.copy2(src_bench, dst_benchmark)
        else:
            print("No existing benchmark_results.json found. The space will use hardcoded fallback metrics.")

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
        shutil.rmtree(TMP_CLONE_DIR, ignore_errors=True)
        sys.exit(0)

    success, _ = run_cmd(["git", "commit", "-m", "Deploy app_gradio to Hugging Face Spaces"], cwd=TMP_CLONE_DIR)
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
    shutil.rmtree(TMP_CLONE_DIR, ignore_errors=True)
    print("Done!")

if __name__ == "__main__":
    main()
