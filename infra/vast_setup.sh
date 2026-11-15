#!/usr/bin/env bash
# Set up (or restart) the demo backend on a Vast.ai GPU machine. Safe to run again after a Stop/Start.
#
#   1. Rent 1x RTX 3090/4090 (24 GB), ~60 GB disk, an image with CUDA (e.g. vastai/base-image or
#      pytorch/pytorch), On-demand. Connect with SSH.
#   2. Put the code at /workspace/Thesis (git clone, or from your PC: scp -P <port> -r Thesis root@<ip>:/workspace/)
#      and the filled-in .env at /workspace/Thesis/.env (scp it; never commit it).
#   3. bash /workspace/Thesis/infra/vast_setup.sh
#
# It installs FFmpeg, the backend venv, Ollama + the LLM, starts the backend on :8001 and a temporary
# Cloudflare tunnel, and writes the tunnel URL to Supabase app_config.backend_url, where the Vercel
# frontend reads it. Logs: /workspace/logs/*.log
set -euo pipefail

ROOT=${ROOT:-/workspace/Thesis}
LOGS=/workspace/logs
MODEL=${OLLAMA_MODEL:-qwen2.5:7b}
mkdir -p "$LOGS"
cd "$ROOT"
[ -f .env ] || { echo "Thiếu $ROOT/.env (chép từ máy bạn, theo .env.example)"; exit 1; }
set -a; source .env; set +a

echo "== 1/5 gói hệ thống"
if ! command -v ffmpeg >/dev/null || ! command -v curl >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg curl python3-venv >/dev/null
fi

echo "== 2/5 backend (Python venv)"
if [ ! -x backend/.venv/bin/python ]; then python3 -m venv backend/.venv; fi
backend/.venv/bin/pip install -q --upgrade pip
backend/.venv/bin/pip install -q -r backend/requirements.txt

echo "== 3/5 Ollama + $MODEL"
if ! command -v ollama >/dev/null; then curl -fsSL https://ollama.com/install.sh | sh >/dev/null; fi
if ! curl -s -m 2 localhost:11434/api/tags >/dev/null; then
  OLLAMA_KEEP_ALIVE=24h nohup ollama serve >"$LOGS/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do curl -s -m 2 localhost:11434/api/tags >/dev/null && break; sleep 2; done
fi
ollama pull "$MODEL" >/dev/null
# load the model into GPU memory now, so the first question does not wait for it
curl -s localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"24h\",\"options\":{\"num_predict\":1}}" >/dev/null

echo "== 4/5 backend :8001"
pkill -f "uvicorn app.main:app" 2>/dev/null || true
(cd backend && LLM_PROVIDER=${LLM_PROVIDER:-ollama} OLLAMA_MODEL="$MODEL" \
  nohup .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 >"$LOGS/backend.log" 2>&1 &)
for _ in $(seq 1 60); do curl -s -m 2 localhost:8001/api/health >/dev/null && break; sleep 2; done
curl -s localhost:8001/api/health; echo
(cd backend && .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('../.env')
from app.shared import db
missing = db.check_schema() if db.enabled() else ['SUPABASE_DB_URL chưa đặt']
print('Supabase:', 'sẵn sàng' if not missing else f'thiếu {missing}')
")

echo "== 5/5 Cloudflare tunnel (link tạm)"
if [ ! -x /usr/local/bin/cloudflared ]; then
  curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
pkill -f "cloudflared tunnel" 2>/dev/null || true
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8001 >"$LOGS/tunnel.log" 2>&1 &
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOGS/tunnel.log" | head -1 || true)
  [ -n "$URL" ] && break; sleep 2
done
[ -n "$URL" ] || { echo "Không lấy được link tunnel, xem $LOGS/tunnel.log"; exit 1; }
(cd backend && .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('../.env')
from app.shared import db
db.set_config('backend_url', '$URL'); print('Đã ghi app_config.backend_url =', '$URL')
")
echo
echo "Xong. Backend: $URL/api/health   ·   mở link Vercel để dùng."
echo "Tắt: Stop máy trên Vast (giữ ổ đĩa). Bật lại: chạy lại script này."
