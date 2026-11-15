# Demo tất cả trên một máy Vast.ai

Máy GPU chạy pipeline trích xuất, Ollama, FastAPI và frontend. Dữ liệu nằm ở `/workspace/app-data`; không cần Vercel, Supabase hay R2.

## 1. Thuê máy

Chọn Ubuntu có CUDA, GPU RTX 3090/4090 với 24 GB VRAM, khoảng 80 GB ổ đĩa và kết nối SSH. Dùng ảnh CUDA tương thích với `pipeline/requirements-gpu.txt`. Chọn On-demand cho buổi demo; giữ ổ đĩa khi Stop. Script cần quyền root và Internet để tải các gói và mô hình.

## 2. Chép mã, dữ liệu và cấu hình

Chuẩn bị ba tệp trên máy cá nhân:

- `thesis-code.tar.gz`: mã nguồn, gồm `backend/`, `frontend/`, `infra/`, `pipeline/` và các tài nguyên pipeline cần; loại bỏ `.git`, các venv, `node_modules`, `frontend/dist`, `.env`, dữ liệu và thư mục chạy pipeline cũ.
- `app-data.tar.gz`: đóng gói **nội dung** thư mục dữ liệu hiện tại (khoảng 1,4 GB), gồm `videos/`, `transcripts/`, `keyframes/`, `slide_variants/` và `index/`.
- `.env`: bản riêng cho demo, **bỏ toàn bộ khóa Kaggle** (`KAGGLE_USERNAME`, `KAGGLE_KEY` và các khóa liên quan). Không chép tệp thông tin đăng nhập khác. Đặt `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=qwen2.5:7b`; hoặc dùng `LLM_PROVIDER=gemini` cùng khóa Gemini của bạn. Để trống mọi biến `SUPABASE_*` và `R2_*`; script cũng ép các biến này về rỗng.

Ví dụ từ thư mục chứa ba tệp, thay cổng và IP bằng thông tin SSH trên Vast.ai:

```bash
scp -P <port> thesis-code.tar.gz app-data.tar.gz .env root@<ip>:/workspace/
ssh -p <port> root@<ip>
mkdir -p /workspace/Thesis /workspace/app-data
tar -xzf /workspace/thesis-code.tar.gz -C /workspace/Thesis
tar -xzf /workspace/app-data.tar.gz -C /workspace/app-data
install -m 600 /workspace/.env /workspace/Thesis/.env
```

Hai tarball nên chứa các thư mục nêu trên ngay ở cấp đầu, không lồng thêm một thư mục gốc. Không in nội dung `.env` hay đưa khóa vào lịch sử lệnh. Nếu đặt `ADMIN_TOKEN`, các thao tác ghi cần header `X-Admin-Token`; giao diện hiện tại phù hợp buổi demo cá nhân với `ADMIN_TOKEN` để trống. URL tunnel là công khai, chỉ chia sẻ với người tham gia demo.

## 3. Chạy và dùng ứng dụng

```bash
bash /workspace/Thesis/infra/vast_allinone.sh
```

Script có thể chạy lại sau Stop/Start. Nó cài ffmpeg, curl, git, Node.js 20/npm, tạo hai venv riêng ở `backend/.venv` và `backend/.venv-pipeline`, cài Ollama khi dùng provider Ollama, build frontend và mở tunnel. Mở URL `https://…trycloudflare.com` được in cuối lệnh. URL thay đổi mỗi lần khởi động tunnel.

Thêm video bằng liên kết hoặc tải tệp lên. Video tự vào hàng đợi: **Tách slide → Lọc khung → OCR → Nhận dạng giọng nói → Đồng bộ → Tạo chỉ mục → Nạp dữ liệu**. Mỗi lần chỉ một video được trích xuất. Theo dõi tiến độ trong cửa sổ nhập hoặc danh sách video; khi hoàn tất, có thể đặt câu hỏi. Nếu thất bại, dùng nút **Trích xuất video** để thử lại.

Backend lắng nghe ở `127.0.0.1:8001`, phục vụ cả `/api` và frontend. Chỉ chạy **một worker uvicorn** vì hàng đợi nằm trong một tiến trình. Ollama không giữ mô hình trong VRAM sau câu trả lời để dành bộ nhớ cho pipeline. Tránh đặt câu hỏi trong lúc đang trích xuất video lớn nếu GPU thiếu bộ nhớ.

Các biến script đặt: `LECTURE_RAG_DATA_DIR=/workspace/app-data`, `LECTURE_EXTRACT=on`, `PIPELINE_PYTHON=/workspace/Thesis/backend/.venv-pipeline/bin/python`, `SERVE_FRONTEND_DIR=/workspace/Thesis/frontend/dist`. Có thể đặt `LECTURE_EXTRACT_LANGUAGE=vi|en|auto` (mặc định `auto`) và `LECTURE_EXTRACT_DEVICE=cuda|cpu` (mặc định `cuda`) trong `.env`.

Nhật ký dịch vụ và cài đặt ở `/workspace/logs/`; stderr pipeline ở `/workspace/app-data/extract/work/<video_id>/<run_id>/pipeline.stderr.log`. Trạng thái JSON ở `/workspace/app-data/extract/jobs/`. Trạng thái hoàn tất/thất bại được giữ sau restart; công việc đang dở được đánh dấu thất bại để thử lại. Không chia sẻ nhật ký hoặc cấu hình nếu chưa kiểm tra thông tin nhạy cảm.

## 4. Kết thúc demo

**Stop máy trên Vast.ai ngay sau demo** để dừng tính phí GPU; ổ đĩa lưu trữ vẫn có thể phát sinh phí. Không chọn Destroy nếu muốn giữ dữ liệu. Có thể chép dữ liệu mới về máy cá nhân trước khi Destroy. Sau khi Start lại, chạy lại script và mở URL mới.

## Máy 2 GPU

Script tự đếm số GPU:
- **2 GPU trở lên:** trích xuất chạy trên GPU 0 (`EXTRACT_GPU`), Ollama trả lời trên GPU 1 (`OLLAMA_GPU`) và giữ model trong bộ nhớ, nên vừa trích xuất vừa hỏi vẫn nhanh.
- **1 GPU:** dùng chung; Ollama nhả bộ nhớ sau mỗi câu trả lời để nhường chỗ cho trích xuất.

Ngay sau khi cài môi trường pipeline, script chạy thử PyTorch trên GPU trích xuất (in tên card, đời `sm_xx`, dung lượng). Nếu card không chạy được (ví dụ bản PyTorch chưa hỗ trợ) hoặc VRAM dưới 12 GB, script dừng sớm. Khi đó nên Destroy máy và đổi sang RTX 3090/4090.
