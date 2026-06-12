import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

# Queries and expected IDs
QUERIES = [
    ("tiêm vắc xin có làm trẻ bị tự kỷ không", "myth_001"),
    ("chích ngừa covid có gây hiếm muộn không", "myth_003"),
    ("trong vắc xin có con chip điện tử theo dõi", "myth_005"),
    ("thủy ngân trong vắc xin gây hại thần kinh", "myth_007"),
    ("công nghệ mrna sửa đổi gen người", "myth_012"),
    ("bà bầu tiêm phòng có bị sảy thai", "myth_004"),
    ("vắc xin hpv khiến trẻ quan hệ sớm", "myth_016"),
    ("tiêm cúm xong lại bị cúm", "myth_019"),
    ("vắc xin làm cơ thể hút nam châm", "myth_015"),
    ("uống nước gừng sả chanh chữa khỏi covid", "myth_032")
]

def run_worker(model: str, db_path: str):
    # Set environments
    os.environ["EMBEDDING_MODEL"] = model
    os.environ["INDEX_DB_PATH"] = db_path
    
    # Avoid local datasets import conflict by removing empty and dot from sys.path
    sys.path = [p for p in sys.path if p not in ('', '.')]
    
    # Add xai_service to path
    here = Path(__file__).resolve()
    xai_service_path = here.parents[1]
    sys.path.insert(0, str(xai_service_path))
    
    from app.embedding_index import build_index, query, index_stats
    
    # Load fact_kb.json
    kb_path = xai_service_path.parent / "data" / "fact_kb.json"
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)
        
    docs = [
        {
            "id": x["id"],
            "text": x["myth"] + " " + x["fact"],
            "source": x["sources"][0].get("org", "") if x.get("sources") else "",
            "url": x["sources"][0].get("url", "") if x.get("sources") else ""
        }
        for x in kb
    ]
    
    # Build index
    build_index(docs, rebuild=True)
    stats = index_stats()
    
    # Benchmark
    hits = 0
    times = []
    
    # Warmup query
    query("warmup", 5)
    
    for q_text, expected_id in QUERIES:
        t0 = time.perf_counter()
        results = query(q_text, 5)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # in ms
        
        # Check recall@5
        top_ids = [r["id"] for r in results]
        if expected_id in top_ids:
            hits += 1
            
    recall = hits / len(QUERIES)
    avg_time = sum(times) / len(times)
    
    result = {
        "model": model,
        "dim": stats["dim"],
        "recall": recall,
        "avg_time_ms": avg_time
    }
    
    print(json.dumps(result))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--db", type=str)
    args = parser.parse_args()
    
    if args.worker:
        run_worker(args.model, args.db)
        return
        
    # Main orchestrator mode
    models = [
        "intfloat/multilingual-e5-small",
        "dangvantuan/vietnamese-embedding"
    ]
    
    results = []
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())
    
    for i, model in enumerate(models):
        db_name = f"_bench_e5.db" if "e5" in model.lower() else f"_bench_dvt.db"
        db_path = str(Path(__file__).resolve().parents[2] / "data" / db_name)
        
        # Run worker subprocess
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        
        proc = subprocess.run(
            [python_exe, script_path, "--worker", "--model", model, "--db", db_path],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8"
        )
        
        if proc.returncode != 0:
            print(f"Error running bench for {model}:")
            print(proc.stderr)
            sys.exit(1)
            
        # Parse result from stdout (last line)
        stdout_lines = proc.stdout.strip().split("\n")
        # Find json line
        json_line = None
        for line in reversed(stdout_lines):
            if line.strip().startswith("{") and line.strip().endswith("}"):
                json_line = line
                break
        if json_line:
            results.append(json.loads(json_line))
        else:
            print(f"Failed to parse JSON output for {model}. Raw output:")
            print(proc.stdout)
            sys.exit(1)
            
    # Print comparison table
    print("\n" + "="*70)
    print(f"{'Model Name':<40} | {'Dim':<5} | {'Recall@5':<10} | {'Avg Latency (ms)':<15}")
    print("-" * 78)
    for res in results:
        print(f"{res['model']:<40} | {res['dim']:<5} | {res['recall']:<10.2%} | {res['avg_time_ms']:<15.2f}")
    print("="*70)
    
    # Recommendation logic
    best_res = sorted(results, key=lambda x: (-x["recall"], x["avg_time_ms"]))[0]
    print(f"\nKhuyến nghị: Sử dụng mô hình '{best_res['model']}' vì đạt Recall@5 tối ưu ({best_res['recall']:.1%}) với độ trễ thấp ({best_res['avg_time_ms']:.2f} ms).")

if __name__ == "__main__":
    main()
