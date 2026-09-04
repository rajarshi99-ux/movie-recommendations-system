# Run CineMatch Full-Stack App

Write-Host "Starting Backend FastAPI on port 8000..." -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m", "uvicorn", "backend.main:app", "--port", "8000"

Write-Host "Starting Frontend HTTP Server on port 8080..." -ForegroundColor Cyan
Push-Location "frontend"
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m", "http.server", "8080"
Pop-Location

Write-Host ""
Write-Host "Servers started successfully!" -ForegroundColor Green
Write-Host "Backend API:  http://localhost:8000"
Write-Host "Frontend Vue/React/HTML: http://localhost:8080"
Write-Host ""
Write-Host "Press Ctrl+C to terminate the powershell, but you may have to kill python processes manually."
