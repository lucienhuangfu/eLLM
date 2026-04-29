$baseUrl = "http://127.0.0.1:8000"

Write-Host "[1/3] 检查服务状态..."
curl.exe "$baseUrl/status"

Write-Host "`n[2/3] 测试非流式 chat/completions..."
curl.exe -X POST "$baseUrl/v1/chat/completions" `
  -H "Content-Type: application/json" `
  -d '{"model":"local-model","messages":[{"role":"user","content":"你好，请简单介绍一下你自己"}],"stream":false}'

Write-Host "`n[3/3] 测试流式 chat/completions (SSE)..."
curl.exe -N -X POST "$baseUrl/v1/chat/completions" `
  -H "Content-Type: application/json" `
  -d '{"model":"local-model","messages":[{"role":"user","content":"请用两句话介绍 Rust"}],"stream":true}'
