#!/usr/bin/env bash
# Ubuntu/CUDA, one GPU host for extraction, local data, API and frontend.
set +x
set -euo pipefail
umask 077

ROOT=${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
LOGS=/workspace/logs
mkdir -p "$LOGS" /workspace/app-data
cd "$ROOT"
trap 'echo "Thiết lập chưa hoàn tất. Xem nhật ký trong /workspace/logs." >&2' ERR

echo "1/7: Chuẩn bị ffmpeg, curl, git, Python và Node.js 20"
if ! command -v ffmpeg >/dev/null || ! command -v curl >/dev/null || \
   ! command -v git >/dev/null || ! dpkg-query -W python3-venv >/dev/null 2>&1; then
  apt-get update >>"$LOGS/setup.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg curl git python3-venv ca-certificates >>"$LOGS/setup.log" 2>&1
fi
if ! command -v node >/dev/null || [[ $(node -p 'process.versions.node.split(".")[0]') != 20 ]] || ! command -v npm >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x -o "$LOGS/nodesource-setup.sh"
  bash "$LOGS/nodesource-setup.sh" >>"$LOGS/setup.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs >>"$LOGS/setup.log" 2>&1
fi

echo "2/7: Chuẩn bị môi trường backend"
[[ -x backend/.venv/bin/python ]] || python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt >>"$LOGS/setup.log" 2>&1

# Read dotenv as data, rather than executing shell code. Values are never printed.
# Python emits safely quoted exports into a private temporary file, then removes it.
ENV_FILE=$(mktemp "$LOGS/.env-load.XXXXXX")
backend/.venv/bin/python - "$ENV_FILE" <<'PY'
import re
import shlex
import sys
from pathlib import Path
from dotenv import dotenv_values
values = dotenv_values('.env') if Path('.env').is_file() else {}
with open(sys.argv[1], 'w', encoding='utf-8') as output:
    for key, value in values.items():
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) and value is not None:
            output.write(f'export {key}={shlex.quote(value)}\n')
PY
source "$ENV_FILE"
rm -f -- "$ENV_FILE"

# Empty exported values prevent backend load_dotenv from restoring cloud settings.
for name in $(compgen -v); do
  case "$name" in SUPABASE_*|R2_*|VITE_SUPABASE_*) export "$name=" ;; esac
done
export SUPABASE_DB_URL= SUPABASE_URL= SUPABASE_ANON_KEY= SUPABASE_SERVICE_ROLE_KEY=
export R2_ACCOUNT_ID= R2_ACCESS_KEY_ID= R2_SECRET_ACCESS_KEY= R2_BUCKET= R2_ENDPOINT= R2_REGION=
export VITE_SUPABASE_URL= VITE_SUPABASE_ANON_KEY= VITE_API_BASE_URL=/api
export LECTURE_RAG_DATA_DIR=/workspace/app-data
export LECTURE_EXTRACT=on SERVE_FRONTEND_DIR="$ROOT/frontend/dist"
export PIPELINE_PYTHON="$ROOT/backend/.venv-pipeline/bin/python"
export LECTURE_EXTRACT_LANGUAGE=${LECTURE_EXTRACT_LANGUAGE:-auto}
export LECTURE_EXTRACT_DEVICE=${LECTURE_EXTRACT_DEVICE:-cuda}
export LLM_PROVIDER=${LLM_PROVIDER:-ollama}
export OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5:7b}
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://127.0.0.1:11434}

# GPU layout: with 2+ GPUs, extraction uses EXTRACT_GPU and Ollama uses OLLAMA_GPU, so answering
# stays fast while a video is extracted. With 1 GPU they share it and Ollama unloads between answers.
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >>"$LOGS/setup.log" 2>&1 || true
if (( GPU_COUNT >= 2 )); then
  export EXTRACT_GPU=${EXTRACT_GPU:-0} OLLAMA_GPU=${OLLAMA_GPU:-1}
  export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-24h}
  echo "Phát hiện $GPU_COUNT GPU: trích xuất dùng GPU $EXTRACT_GPU, trả lời dùng GPU $OLLAMA_GPU."
else
  export EXTRACT_GPU=${EXTRACT_GPU:-0} OLLAMA_GPU=${OLLAMA_GPU:-0}
  # Do not keep the Q&A model resident while extraction uses GPU memory.
  export OLLAMA_KEEP_ALIVE=0
  echo "Phát hiện $GPU_COUNT GPU: trích xuất và trả lời dùng chung."
fi

echo "3/7: Chuẩn bị môi trường pipeline GPU"
[[ -f pipeline/requirements-gpu.txt ]] || { echo "Thiếu pipeline/requirements-gpu.txt." >&2; exit 1; }
[[ -x "$PIPELINE_PYTHON" ]] || python3 -m venv "$ROOT/backend/.venv-pipeline"
"$PIPELINE_PYTHON" -m pip install -r pipeline/requirements-gpu.txt >>"$LOGS/setup.log" 2>&1
# Fail early when this PyTorch build cannot run kernels on the card (e.g. new Blackwell GPUs).
if [[ "$LECTURE_EXTRACT_DEVICE" == cuda ]]; then
  CUDA_VISIBLE_DEVICES=$EXTRACT_GPU "$PIPELINE_PYTHON" - <<'PY' || { echo "PyTorch không chạy được trên GPU này. Xem /workspace/logs/setup.log; nên đổi máy (RTX 3090/4090)." >&2; exit 1; }
import torch
assert torch.cuda.is_available(), "CUDA not available"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
x = torch.ones(1024, 1024, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); float((x @ x).sum())
gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU OK: {name} sm_{cap[0]}{cap[1]} {gb:.1f} GiB, torch {torch.__version__}")
assert gb >= 11, "Need >= 12 GB VRAM for Qwen3-VL-4B OCR"
PY
fi

echo "4/7: Chuẩn bị mô hình trả lời"
case "$LLM_PROVIDER" in
  ollama)
    if ! command -v ollama >/dev/null; then
      curl -fsSL https://ollama.com/install.sh -o "$LOGS/ollama-install.sh"
      bash "$LOGS/ollama-install.sh" >>"$LOGS/setup.log" 2>&1
    fi
    if ! curl -fsS -m 2 "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES=$OLLAMA_GPU nohup ollama serve >"$LOGS/ollama.log" 2>&1 &
      echo $! >"$LOGS/ollama.pid"
      for _ in $(seq 1 60); do
        curl -fsS -m 2 "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1 && break
        sleep 1
      done
    fi
    curl -fsS -m 2 "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1
    ollama pull "$OLLAMA_MODEL" >>"$LOGS/setup.log" 2>&1
    ;;
  gemini) ;; # Uses GEMINI_API_KEY from .env; no local model needed.
  *) echo "LLM_PROVIDER phải là ollama hoặc gemini." >&2; exit 1 ;;
esac

echo "5/7: Biên dịch frontend"
(cd frontend && npm ci && npm run build) >>"$LOGS/build.log" 2>&1

# Only stop processes launched by this script, checking PID reuse against cmdline.
stop_owned() {
  local pid_file=$1 expected=$2 pid
  if [[ -f "$pid_file" ]]; then
    read -r pid <"$pid_file" || true
    if [[ ${pid:-} =~ ^[0-9]+$ ]] && [[ -r /proc/$pid/cmdline ]] && \
       tr '\0' ' ' </proc/"$pid"/cmdline | grep -Fq -- "$expected"; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
    fi
    rm -f -- "$pid_file"
  fi
}

echo "6/7: Khởi động API và frontend"
stop_owned "$LOGS/backend.pid" "uvicorn app.main:app"
(
  cd backend
  # The backend itself runs on CPU; its extraction subprocesses inherit this GPU.
  CUDA_VISIBLE_DEVICES=$EXTRACT_GPU nohup .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1 >"$LOGS/backend.log" 2>&1 &
  echo $! >"$LOGS/backend.pid"
)
READY=false
for _ in $(seq 1 60); do
  if curl -fsS -m 2 http://127.0.0.1:8001/api/health >/dev/null 2>&1; then READY=true; break; fi
  sleep 1
done
[[ "$READY" == true ]]
kill -0 "$(cat "$LOGS/backend.pid")" 2>/dev/null

echo "7/7: Mở Cloudflare quick tunnel"
if ! command -v cloudflared >/dev/null; then
  case "$(uname -m)" in x86_64) ARCH=amd64 ;; aarch64) ARCH=arm64 ;; *) exit 1 ;; esac
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH" -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi
stop_owned "$LOGS/tunnel.pid" "cloudflared tunnel"
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8001 >"$LOGS/tunnel.log" 2>&1 &
echo $! >"$LOGS/tunnel.pid"
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1 || true)
  [[ -n "$URL" ]] && break
  sleep 1
done
[[ -n "$URL" ]] || { echo "Chưa lấy được URL. Xem /workspace/logs/tunnel.log." >&2; exit 1; }
echo "Mở ứng dụng: $URL"
echo "Sau demo: Stop máy trên Vast.ai. Bật lại máy rồi chạy lại script để có URL mới."
