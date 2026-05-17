@echo off
chcp 65001 >nul
echo ========================================================
echo        THIẾT LẬP VÀ CẬP NHẬT FILE PUSH LÊN GITHUB
echo ========================================================
echo Repository mục tiêu: https://github.com/hwngkm/VaccineNLP-Thesis
echo Thư mục hiện tại: %~dp0
echo.

cd /d "%~dp0"

:: 1. Kiểm tra Git đã được cài đặt chưa
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [LỖI] Git chưa được cài đặt hoặc chưa được thêm vào môi trường (PATH)!
    echo Vui lòng tải và cài đặt Git từ: https://git-scm.com/
    pause
    exit /b
)

:: 2. Khởi tạo Git nếu chưa có
if not exist ".git" (
    echo [INFO] Chưa có Git repository. Đang khởi tạo...
    git init
    git branch -M main
) else (
    echo [INFO] Git đã được khởi tạo trước đó.
)

:: 3. Thiết lập Remote Origin là repository mới
git remote | findstr /i "origin" >nul
if %errorlevel% equ 0 (
    echo [INFO] Đang cập nhật Remote Origin sang: https://github.com/hwngkm/VaccineNLP-Thesis.git
    git remote set-url origin https://github.com/hwngkm/VaccineNLP-Thesis.git
) else (
    echo [INFO] Đang cấu hình Remote Origin: https://github.com/hwngkm/VaccineNLP-Thesis.git
    git remote add origin https://github.com/hwngkm/VaccineNLP-Thesis.git
)

:: 4. Thêm các file thay đổi (bao gồm scripts, app và các thư mục khác)
echo [INFO] Đang thêm các file thay đổi vào Git...
git add .

:: 5. Commit các thay đổi
echo [INFO] Đang tạo bản commit...
git commit -m "Cập nhật scripts và mã nguồn VaccineNLP Thesis"

:: 6. Push lên GitHub
echo [INFO] Đang tiến hành push lên nhánh main của GitHub...
git push -u origin main

:: Nếu push bị từ chối do xung đột lịch sử (ví dụ: repo trên github có sẵn file)
if %errorlevel% neq 0 (
    echo.
    echo [CẢNH BÁO] Có thể có xung đột với lịch sử trên Github (ví dụ repo Github không trống).
    echo Đang thực hiện kéo (pull --rebase) về trước khi push...
    git pull origin main --rebase
    echo.
    echo [INFO] Đang thực hiện push lại lên Github...
    git push -u origin main
)

if %errorlevel% equ 0 (
    echo.
    echo ========================================================
    echo [THÀNH CÔNG] Đã cập nhật và push thành công lên GitHub!
    echo Link repo: https://github.com/hwngkm/VaccineNLP-Thesis
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo [LỖI] Có lỗi xảy ra trong quá trình push. 
    echo Hãy đảm bảo rằng bạn đã đăng nhập Git trên máy hoặc đã có quyền truy cập repo này.
    echo ========================================================
)

pause
