# Script chuẩn hóa cấu trúc dự án VaccineNLP theo bản mẫu chuyên nghiệp
# Được tạo bởi Antigravity AI

$BasePath = "c:\Users\dinhl\Downloads\VaccineNLP_ĐỒ_ÁN"
cd $BasePath

Write-Host "--- Bắt đầu quy trình chuẩn hóa cấu trúc dự án ---" -ForegroundColor Cyan

# 1. Đổi tên các thư mục cơ bản
if (Test-Path "notebook") { 
    Write-Host "Renaming 'notebook' to 'notebooks'..."
    Rename-Item -Path "notebook" -NewName "notebooks" 
}
if (Test-Path "script") { 
    Write-Host "Renaming 'script' to 'scripts'..."
    Rename-Item -Path "script" -NewName "scripts" 
}

# 2. Tạo cấu trúc experiments
if (!(Test-Path "experiments")) { 
    Write-Host "Creating 'experiments' directory structure..."
    mkdir experiments 
}
if (!(Test-Path "experiments/models")) { mkdir experiments/models }
if (!(Test-Path "experiments/results")) { mkdir experiments/results }

# 3. Di chuyển models và output vào experiments
if (Test-Path "models") {
    Write-Host "Moving 'models' to 'experiments/models'..."
    Get-ChildItem -Path "models/*" | Move-Item -Destination "experiments/models" -Force
    Remove-Item "models" -Recurse
}
if (Test-Path "output") {
    Write-Host "Moving 'output' to 'experiments/results'..."
    Get-ChildItem -Path "output/*" | Move-Item -Destination "experiments/results" -Force
    Remove-Item "output" -Recurse
}

# 4. Tạo các thư mục còn thiếu để bám sát bản mẫu
$Folders = @("app", "docs", "scratch")
foreach ($folder in $Folders) {
    if (!(Test-Path $folder)) {
        Write-Host "Creating missing folder: '$folder'..."
        mkdir $folder
    }
}

# 5. Cập nhật lại đường dẫn trong script download_all.ps1 (nếu có)
$DownloadScript = "experiments/results/download_all.ps1"
if (Test-Path $DownloadScript) {
    Write-Host "Updating paths in download_all.ps1..."
    $content = Get-Content $DownloadScript
    $content = $content -replace "output", "experiments/results"
    $content | Set-Content $DownloadScript
}

Write-Host "`n[DONE] Cấu trúc dự án đã được chuẩn hóa!" -ForegroundColor Green
Write-Host "Bạn có thể kiểm tra lại cây thư mục ngay bây giờ."
