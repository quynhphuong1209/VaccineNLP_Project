import os
import json
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
import datetime

class VaccineLLMAnnotator:
    def __init__(self):
        # Load API keys
        self.api_keys = []
        self._load_keys_from_env()
        if not self.api_keys:
            raise ValueError("🚨 Không tìm thấy bất kỳ GEMINI_API_KEY nào trong .env!")
        
        self.current_key_idx = 0
        self.client = self._create_client()
        self.model = "gemma-4-31b-it"

        self.system_instruction = """<|think|>
You are a highly rigorous Public Health Data Analyst in Vietnam. Your task is to analyze social media comments regarding vaccines and annotate them strictly.

CRITICAL INSTRUCTION: Do not be distracted by philosophical arguments, political rhetoric, or manipulative half-truths. Focus entirely on identifying the author's TRUE attitude toward VACCINATION, which is typically revealed in the opening or closing sentences.

Analyze the comment, reason step-by-step, and classify it based on these 3 criteria:
1. Misinformation: [Chính xác] or [Tin giả / Sai lệch] or [Không chắc chắn / Không liên quan]
2. Stance: [Ủng hộ] or [Phản đối] or [Trung lập]
3. Sentiment: [Tích cực] or [Tiêu cực] or [Trung tính]

OUTPUT RULE:
After your reasoning, the VERY LAST LINE of your response MUST exactly match this format:
Kết quả: <Misinformation Label> | <Stance Label> | <Sentiment Label>
"""

        self.config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=1.0,
            top_p=0.95,
            top_k=64
        )

    def _load_keys_from_env(self):
        env_path = '.env'
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('GEMINI_API_KEY'):
                        parts = line.split('=', 1)
                        if len(parts) == 2 and parts[1]:
                            self.api_keys.append(parts[1])

    def _create_client(self):
        """Khởi tạo client với API key hiện tại"""
        # Lưu ý: SDK yêu cầu timeout tính bằng MILI GIÂY (ms)
        # 60000ms = 60s
        return genai.Client(
            api_key=self.api_keys[self.current_key_idx],
            http_options={'timeout': 60000}
        )

    def _rotate_key(self):
        self.current_key_idx += 1
        if self.current_key_idx >= len(self.api_keys):
            print("🚨 Đã hết API keys dự phòng! Vui lòng nạp thêm.")
            return False
        print(f"🔄 Xoay vòng Key... chuyển sang Key thứ {self.current_key_idx + 1}")
        self.client = self._create_client()
        return True

    def annotate(self, text, retries=3):
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=text,
                    config=self.config
                )
                
                raw_response = response.text
                if not raw_response:
                    return None
                 
                # Parsing logic for Gemma 4 <|channel|> structure
                final_answer = raw_response
                thought_process = ""
                
                if "<channel|>" in raw_response:
                    parts = raw_response.split("<channel|>")
                    final_answer = parts[-1].strip()
                    # Extract thought process
                    if "<|channel>thought" in parts[0]:
                        thought_process = parts[0].split("<|channel>thought")[-1].strip()
                else:
                    # Fallback for models without channel tags
                    final_answer = raw_response
                    thought_process = raw_response
                
                # Cleanup special control tokens just in case
                final_answer = final_answer.replace('<turn|>', '').replace('</start_of_turn>', '').strip()
                
                # Find the result line in the final answer
                lines = final_answer.split('\n')
                result_line = None
                for line in reversed(lines):
                    if line.strip().startswith("Kết quả:"):
                        result_line = line.strip()
                        break
                
                parsed_labels = None
                if result_line:
                    segments = result_line.replace("Kết quả:", "").split("|")
                    if len(segments) == 3:
                        parsed_labels = {
                            "Misinformation": segments[0].strip(),
                            "Stance": segments[1].strip(),
                            "Sentiment": segments[2].strip()
                        }
                
                return {
                    "raw_output": raw_response,
                    "thought_process": thought_process,
                    "result_line": result_line,
                    "parsed_labels": parsed_labels
                }

            except APIError as e:
                # Catch rate limits, server errors, or timeouts
                err_str = str(e).lower()
                is_retryable = any(code in err_str for code in ['429', '500', '503', '504', 'timeout', 'deadline', 'user_rate_limit'])
                
                if is_retryable:
                    print(f"⚠️ Lỗi API/Timeout ({e}). Đang xoay vòng key...")
                    time.sleep(2) # Cooldown before rotation
                    if self._rotate_key():
                        continue
                    else:
                        break
                else:
                    print(f"Lỗi API khác: {e}")
                    break
            except Exception as e:
                err_str = str(e).lower()
                if "timeout" in err_str or "deadline" in err_str or "stream" in err_str:
                    print(f"🔄 Phát hiện timeout ngầm ({e}), đang xoay vòng key...")
                    time.sleep(3)
                    if self._rotate_key():
                        continue
                print(f"Lỗi không xác định: {e}")
                time.sleep(5)
                continue
                
        return None

    def process_batch(self, input_file, output_file):
        print(f"🚀 Bắt đầu LLM Inference bằng {self.model} via google-genai SDK")
        print(f"🔑 Đã nạp {len(self.api_keys)} API keys.")
        
        if not os.path.exists(input_file):
            print(f"❌ File input không tồn tại: {input_file}")
            return
            
        with open(input_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
            
        print(f"📦 Đã load {len(corpus)} items từ {input_file}")
        
        processed_ids = set()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        processed_ids.add(record.get('text'))
                    except json.JSONDecodeError:
                        pass
            print(f"🔄 Checkpoint: Đã load {len(processed_ids)} items đã xử lý từ jsonl.")

        new_items = 0
        with open(output_file, 'a', encoding='utf-8') as f:
            for i, item in enumerate(corpus):
                text = item.get('text', '')
                if text in processed_ids:
                    continue
                    
                start_time = datetime.datetime.now()
                timestamp = start_time.strftime("%H:%M:%S")
                print(f"[{timestamp}] [{i+1}/{len(corpus)}] Đang phân tích...", end=" ", flush=True)
                
                while True:
                    annotation = self.annotate(text)
                    
                    if annotation:
                        end_time = datetime.datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        print(f"✅ Xong ({duration:.1f}s)")
                        # Ghi thêm vào bản record
                        item['llm_thought'] = annotation['thought_process']
                        item['llm_raw_output'] = annotation['raw_output']
                        item['llm_parsed_labels'] = annotation['parsed_labels']
                        item['status'] = 'llm_annotated'
                        
                        # Cập nhật checkpoint từng dòng (JSONL)
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        f.flush()
                        processed_ids.add(text)
                        new_items += 1
                        break
                    else:
                        print(f"\n🚨 CẢ {len(self.api_keys)} KEYS ĐỀU THẤT BẠI cho mục này (Có thể do lỗi mạng toàn cục).")
                        print("⏳ Tạm dừng 5 phút trước khi thử lại bản ghi này để chờ mạng ổn định...")
                        time.sleep(300)
                        start_time = datetime.datetime.now() # Reset timer for the retry
                        print(f"[{start_time.strftime('%H:%M:%S')}] Thử lại [{i+1}/{len(corpus)}]...")
                    
                time.sleep(0.5) # Prevent spamming
                
        print(f"✨ Hoàn tất Inference Batch! Đã gán nhãn thêm {new_items} items.")
        print(f"💾 Các items hoàn chỉnh đã lưu dạng JSONL tại: {output_file}")
