# VaccineNLP Final Reorganize & Push Script (v2.0)
$RepoPath = "c:\Users\dinhl\Downloads\VaccineNLP_ĐỒ_ÁN"
cd $RepoPath

Write-Host "--- Cleaning Git Index ---" -ForegroundColor Yellow
# Unstage everything to apply the new ULTRA-CLEAN .gitignore
git rm -r --cached . | Out-Null

Write-Host "--- Staging Cleaned Files ---" -ForegroundColor Green
git add .

# Check size of staged files (approximate)
$stagedSize = (git count-objects -v | Select-String "size-pack" | ForEach-Object { $_.ToString().Split(":")[1].Trim() })
Write-Host "Estimated staged size: $stagedSize KB" -ForegroundColor Cyan

Write-Host "--- Committing (Amending) ---" -ForegroundColor Green
git commit --amend -m "Final Reorganization: VaccineNLP Project Structure (Ultra Clean Code-Only)"

Write-Host "--- Pushing to GitHub ---" -ForegroundColor Cyan
# Increase buffer size for large pushes just in case
git config http.postBuffer 524288000
git push -u origin main -f

Write-Host "`n[DONE] Project reorganized and pushed to: https://github.com/quynhphuong1209/VaccineNLP_Project.git" -ForegroundColor Green
