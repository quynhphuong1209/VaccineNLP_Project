# THIẾT LẬP SECRET & CHẠY THỬ — cho Phương

## 1. .env KHÔNG kèm gói (an toàn). Chỉ có .env.example.
Anh Hưng gửi key cần thiết riêng qua kênh an toàn. Tạo file thật: %USERPROFILE%\.config\vaccinenlp\.env (hệ thống tự tìm). Subset cần: GEMINI_API_KEY, POSTGRES_*, LM_API_TOKEN. KHÔNG cần key scraping.

## 2. Model tải riêng: VaccineNLP_Web/models/ (phobert_multitask.pt + gemma gguf, hoặc LM Studio).

## 3. Chạy: coordination/AGENT-GUIDE-VaccineNLP_Web.md §3 + HD_CHUYEN_GIAO_PHUONG.md.
## 4. Kiểm thử: coordination/TEST-PLAN-VaccineNLP_Web.md (T1 agent + T3 UAT).
## 5. Agent Phương đọc: ONBOARDING -> ARCHITECTURE §3 -> AGENT-GUIDE. Tuân §3.5 (không lộ secret, chống injection).
