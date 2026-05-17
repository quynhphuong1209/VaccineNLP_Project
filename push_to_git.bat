@echo off
chcp 65001 > nul
title VaccineNLP - Git Auto Push Helper
color 0b

echo ============================================================
echo   🧪 VaccineNLP Git Push Helper 🧪
echo ============================================================
echo   Đang kiểm tra trạng thái Git trong thư mục hiện tại...
echo.

git status

echo.
echo ============================================================
set /p CHOICE="👉 Bạn có muốn commit và push toàn bộ thay đổi (bao gồm cả datasets & app/streamlit_demo.py) lên GitHub không? (Y/N): "

if /i "%CHOICE%" neq "Y" (
    echo.
    echo ❌ Đã hủy thao tác push.
    goto end
)

echo.
set /p COMMIT_MSG="👉 Nhập thông điệp commit (Bấm Enter để dùng mặc định: 'update: datasets and huggingface links in streamlit demo'): "

if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=update: datasets and huggingface links in streamlit demo
)

echo.
echo 📦 Đang stage các file thay đổi (git add .)...
git add .

echo.
echo 📝 Đang tạo commit với thông điệp: "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%"

echo.
echo 🚀 Đang đẩy lên remote repository (GitHub)...
git push

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo   🎉 CẬP NHẬT LÊN GITHUB THÀNH CÔNG!
    echo ============================================================
) else (
    echo.
    echo   ❌ Đã xảy ra lỗi trong quá trình push. 
    echo   Vui lòng kiểm tra lại kết nối mạng hoặc quyền truy cập repository.
)

:end
echo.
echo Nhấn phím bất kỳ để thoát...
pause > nul
