# Báo cáo thử nghiệm tăng tốc bước 02 (OCR), ngày 12/10/2026

## Mục tiêu
Một video 30 phút mất 15–20 phút để trích xuất, trong đó bước 02 (OCR) chiếm ~86%. Thử 8 cách tăng tốc trên 3 video L25 và đo cả tốc độ lẫn chất lượng chữ.

## Thiết lập
- **Video:**
  - L25_V017: 19 phút, 18 slide cần OCR;
  - L25_V035: 22 phút, 17 slide;
  - L25_V077: 30 phút, 21 slide.
- **Máy:** Kaggle, 2 × T4. Mỗi video một kernel, chạy song song: `l25-speed-bench3-{v017,v035,v077}-20261012`.
- **Điều kiện chung:** mọi bản đều chạy với cache trống và cùng bộ gác OCR (SigLIP 0,15). Model được nạp trước khi bấm giờ.
- **Thước đo:**
  - thời gian bước 02;
  - độ giống chữ so với bản hiện tại (difflib trên chữ OCR và chữ của vùng hình);
  - tỉ lệ từ của bản hiện tại còn giữ được.
- **Mã:** `kaggle/lecture_rag/build_speed_bench3.py`. Kết quả thô nằm ở `D:/thesis-data/speed-bench3/<video>/speed_bench3.json`.

## Kết quả (tổng 3 video)

| # | Cách | Tổng thời gian | Nhanh hơn | Chữ giống bản cũ | Giữ từ |
|---|---|---|---|---|---|
| – | **base** (pipeline hiện tại) | 1522 s | ×1,00 | 100% | 100% |
| ①② | Dùng lại slide gần trùng + bỏ vùng hình nhỏ/logo | 1460 s | ×1,04 | 100% | 100% |
| – | Vòng lặp riêng, cùng logic với ①② (để kiểm tra) | 1383 s | ×1,10 | 100% | 100% |
| ③ | + gọi Qwen theo lô cho các vùng hình của một slide | 1330 s | ×1,14 | 99,2% | 100% |
| ④ | + chạy chồng: Unlimited-OCR (GPU 0) đọc slide sau trong khi Qwen (GPU 1) đọc slide trước | **1158 s** | **×1,31** | **99,2%** | **100%** |
| ⑦a | ④ + Unlimited-OCR ở cỡ 768 thay cho 1024 | 5511 s | ×0,28 (chậm hơn) | 34,7% | 62,9% |
| ⑤ | Chỉ dùng Qwen3-VL đọc cả slide | 708 s | ×2,15 | 51,9% | 76,3% |
| ⑥ | Chỉ dùng GOT-OCR 2.0 đọc cả slide | 455 s | ×3,35 | 20,3% | 39,9% |
| ⑦b | Unlimited-OCR chạy fp16 | lỗi `masked_scatter_` (Half/Float), không chạy được | | | |

### Theo từng video (số lần nhanh hơn so với base)
| Cách | V017 | V035 | V077 |
|---|---|---|---|
| ①② | 1,02 | 1,05 | 1,06 |
| ③ batch | 1,17 | 1,05 | 1,19 |
| ④ chồng 2 GPU | 1,38 | 1,15 | 1,39 |
| ⑤ chỉ Qwen | 1,78 | 2,50 | 2,33 |
| ⑥ chỉ GOT | 4,10 | 4,29 | 2,58 |

## Nhận xét

1. **③ + ④ là cách tốt nhất trong nhóm giữ nguyên chất lượng: nhanh hơn ~31% (tới 39% với video nhiều hình).**
   - Chữ OCR giữ đúng 100% từ.
   - Độ giống chữ là 99,2%. Phần khác nằm ở **câu mô tả hình** do Qwen sinh ra: khi chạy theo lô, cách diễn đạt thay đổi nhẹ. Ví dụ "khoảng tăng giá trị được đánh dấu là ΔA" thành "đoạn dao động từ giá trị thấp đến cao được đánh dấu là ΔA". Dữ liệu biểu đồ (JSON) giữ nguyên.
2. **①② chỉ tiết kiệm 4%.**
   - Số lần gọi Qwen giảm rõ: 80 → 61 trên 3 video. Riêng V077 giảm từ 35 xuống 23.
   - Nhưng thời gian chủ yếu nằm ở Unlimited-OCR (khoảng 10–13 giây mỗi slide), không phải ở Qwen.
   - Cả 3 video đều không có slide nào trùng nhau, nên chưa thấy được lợi ích của việc dùng lại slide gần trùng. Cách này chỉ có ích với video có quảng cáo chen giữa hoặc giảng viên quay lại slide cũ.
   - Không làm hại chất lượng, nên vẫn có thể bật.
3. **Không giảm cỡ ảnh xuống 768.** Unlimited-OCR rơi vào vòng lặp sinh chữ đến hết giới hạn token. Kết quả là chậm hơn 3,6 lần và mất gần 40% số từ.
4. **Chỉ dùng Qwen3-VL (⑤) nhanh gấp 2,15 lần. Chất lượng của cách này chưa thể kết luận là kém hơn.**
   - Độ giống thấp (52%) phần lớn do khác định dạng: `$...$` so với `\( ... \)`, mô tả hình gộp chung vào chữ.
   - Trên slide toán V017, Qwen còn **đọc đúng hơn** bản hiện tại: "chứa d" (bản cũ ghi "chữa d"), "cos AIB" (bản cũ "ATB"), "(P)" (bản cũ "⟨P⟩"). Qwen cũng đọc cả các đáp án A–D mà bản cũ thiếu.
   - Ngược lại, Qwen mất vị trí của từng khối chữ (bbox) và không tách riêng vùng bảng, biểu đồ.
   - **Cần đo CER trên gold (M3AV + gold set 2) mới kết luận được.**
5. **GOT-OCR 2.0 (⑥) nhanh nhất (×3,35) nhưng không dùng được cho tiếng Việt.** Dấu bị mất hoặc sai gần hết, ví dụ "Trong khóng gian voi hé toa dō", "duròng thāng". Công thức LaTeX thì đọc tốt.
6. **fp16 vẫn lỗi** ở `masked_scatter_` trong code của Unlimited-OCR, giống lần trước. Muốn dùng phải sửa code của model.

## Đề xuất
1. **Đưa ③ + ④ vào pipeline chính,** bật ①② cùng lúc vì không mất gì.
   - Kỳ vọng: video 30 phút giảm từ ~10 phút OCR xuống ~7 phút.
   - Việc cần làm: chuyển hàm `run_custom` (đọc hai lượt, gọi Qwen theo lô, chạy chồng 2 GPU) vào `visual.py`, đặt sau công tắc mới `hybrid_pipelined=True`, rồi thêm test.
2. **Thử nghiệm tiếp ⑤ (chỉ Qwen3-VL):** đo CER và WER trên M3AV và gold set 2, so với hybrid.
   - Nếu CER không tệ hơn, có thể dùng Qwen cho slide chỉ có chữ để nhanh gấp ~2 lần.
   - Slide có bảng hoặc biểu đồ vẫn giữ hybrid.
3. **Bỏ** các hướng 768, GOT và fp16.

## Giới hạn của thử nghiệm
- Chỉ có 3 video, 56 slide, đều thuộc L25 (tiếng Việt, kênh Thanh Niên).
- "Chữ giống bản cũ" lấy bản hiện tại làm mốc, không phải đáp án đúng. Bản hiện tại cũng có lỗi.
- Thời gian base lần này (500 s cho V017) thấp hơn lần chạy pipeline đầy đủ (992 s), vì lần trước còn tính thời gian nạp model và các bước khác chạy cùng. Các tỉ lệ trong báo cáo chỉ so giữa các bản trong cùng một kernel.
