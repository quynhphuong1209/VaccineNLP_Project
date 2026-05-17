$OutputEncoding = [System.Text.Encoding]::UTF8
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  🧪 VaccineNLP Git Push Helper (PowerShell) 🧪" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Đang kiểm tra trạng thái Git trong thư mục hiện tại..."
Write-Host ""

git status

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
$CHOICE = Read-Host "👉 Bạn có muốn commit và push toàn bộ thay đổi (bao gồm cả datasets & app/streamlit_demo.py) lên GitHub không? (Y/N)"

if ($CHOICE -ne "Y" -and $CHOICE -ne "y") {
    Write-Host ""
    Write-Host "❌ Đã hủy thao tác push." -ForegroundColor Red
    Write-Host ""
    Read-Host "Nhấn Enter để thoát..."
    exit
}

Write-Host ""
$COMMIT_MSG = Read-Host "👉 Nhập thông điệp commit (Bấm Enter để dùng mặc định: 'update: datasets and huggingface links in streamlit demo')"

if ([string]::IsNullOrWhiteSpace($COMMIT_MSG)) {
    $COMMIT_MSG = "update: datasets and huggingface links in streamlit demo"
}

Write-Host ""
Write-Host "📦 Đang stage các file thay đổi (git add .)..." -ForegroundColor Yellow
git add .

Write-Host ""
Write-Host "📝 Đang tạo commit với thông điệp: `"$COMMIT_MSG`"" -ForegroundColor Yellow
git commit -m $COMMIT_MSG

Write-Host ""
Write-Host "🚀 Đang đẩy lên remote repository (GitHub)..." -ForegroundColor Yellow
git push

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  🎉 CẬP NHẬT LÊN GITHUB THÀNH CÔNG!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  ❌ Đã xảy ra lỗi trong quá trình push." -ForegroundColor Red
    Write-Host "  Vui lòng kiểm tra lại kết nối mạng hoặc quyền truy cập repository." -ForegroundColor Red
}

Write-Host ""
Read-Host "Nhấn Enter để thoát..."
