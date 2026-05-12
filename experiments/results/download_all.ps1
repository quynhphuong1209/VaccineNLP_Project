# Script tải toàn bộ experiments/results từ Kaggle cho dự án VaccineNLP
# Được tạo bởi Antigravity

# Di chuyển vào thư mục experiments/results nếu đang ở ngoài
$experiments/resultsDir = "c:\Users\dinhl\Downloads\VaccineNLP_ĐỒ_ÁN\experiments/results"
if (!(Test-Path $experiments/resultsDir)) { mkdir $experiments/resultsDir }
cd $experiments/resultsDir

# Tạo các thư mục con để chứa experiments/results riêng biệt (không đè lên nhau)
mkdir -Force 01-phobert, xlm-r-v1, phobert-v2, eval-final, gemma, gemma-qlora

# 1. Tải PhoBERT Multitask Training
Write-Host "--- Downloading 1/6: 01-phobert-multitask-training ---" -ForegroundColor Cyan
kaggle kernels experiments/results inhlqunhphng/01-phobert-multitask-training -p ./01-phobert

# 2. Tải XLM-R V1 Multitask Classifier
Write-Host "--- Downloading 2/6: vaccinenlp-xlm-r-v1-multitask-classifi ---" -ForegroundColor Cyan
kaggle kernels experiments/results inhlqunhphng/vaccinenlp-xlm-r-v1-multitask-classifi -p ./xlm-r-v1

# 3. Tải PhoBERT V2 Multitask Classifier
Write-Host "--- Downloading 3/6: vaccinenlp-phobert-v2-multitask-classifier ---" -ForegroundColor Cyan
kaggle kernels experiments/results inhlqunhphng/vaccinenlp-phobert-v2-multitask-classifier -p ./phobert-v2

# 4. Tải Eval Final T4
Write-Host "--- Downloading 4/6: vaccine-nlp-eval-final-t4 ---" -ForegroundColor Cyan
kaggle kernels experiments/results inhlqunhphng/vaccine-nlp-eval-final-t4 -p ./eval-final

# 5. Tải Gemma E4B IT
Write-Host "--- Downloading 5/6: gemma-e4b-it ---" -ForegroundColor Cyan
kaggle kernels experiments/results inhlqunhphng/gemma-e4b-it -p ./gemma

# 6. Tải Gemma 4B QLoRA Training (Mới bổ sung)
Write-Host "--- Downloading 6/6: 02-gemma4-4b-qlora-training ---" -ForegroundColor Cyan
kaggle kernels experiments/results inhlqunhphng/02-gemma4-4b-qlora-training -p ./gemma-qlora

Write-Host "`n[DONE] Toàn bộ dữ liệu đã được tải về thư mục: $experiments/resultsDir" -ForegroundColor Green
