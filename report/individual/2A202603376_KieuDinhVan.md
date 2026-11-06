# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603376
- **Họ và tên:** Kiều Đình Văn
- **SEED** (in ở ô đầu notebook): 9625

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách | F1 test (775 tin) | F1 SUB (40 tin) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.874 | 0.824 | 0.107 | 0 |
| 2. BoW + MLP | 0.941 | 0.919 | 0.617 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.940 | 0.974 | 65.7 | 0 |
| 3b. Encoder (MiniLM) + LogReg | **0.967** | 1.000 | 37.7 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.957 | 1.000 | 59.5 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — (chỉ chạy SUB) | 1.000 | 1529.4 | 0.008 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — | 1.000 | 1057.5 | 0.075 |
| 4c. LLM nhỏ few-shot | — | 1.000 | 1236.8 | 0.018 |

Nhận xét: từ embedding trở lên, chất lượng trên SUB gần như hoà nhau (chênh dưới 0.05, tức 1–2 tin). Khác biệt thật nằm ở độ trễ và chi phí: LLM chậm hơn 3b khoảng 30–40 lần và tốn tiền. Hàng LLM chỉ đo trên 40 tin nên chỉ so sánh được qua cột F1 SUB.

## 3. Dự đoán và thực tế

1. **`ml_lr` — learning rate = 50.** Mình đoán loss sẽ dao động lởm chởm suốt quá trình train. Thực tế loss chỉ vọt lên khoảng 2.5 ở vòng đầu rồi rơi nhanh xuống 0.005, thấp hơn cả lr = 0.5. **Vì sao lệch:** bước đầu tiên làm b bị trừ khoảng 19 (= 50 × 0.377), mô hình đoán mọi tin là ham với độ tự tin rất cao nên loss tăng vọt. Sau đó loss của logistic regression có dạng cái bát (chỉ một đáy) và gradient của từng từ nhỏ (chia cho n, mỗi từ chỉ xuất hiện trong ít tin), nên mô hình tự sửa được. Mình đã đánh giá thấp việc "lr lớn bao nhiêu là quá lớn" phụ thuộc vào bài toán.

2. **`emb_tricky` — tin nhắn ngụy trang.** Mình đoán Encoder + LogReg sẽ thắng rõ BoW. Thực tế hoà 4/6 – 4/6. Encoder bắt được "brand new phone at no cost" (hiểu "no cost" = free), nhưng lại bỏ sót "complimentary holiday", tin mà BoW bắt được nhờ các từ `mobile`, `reply`. Cả hai đều bỏ sót tin lừa đảo tài khoản "account rewards expire". **Vì sao lệch:** encoder chỉ học spam SMS kiểu cũ trong dữ liệu train, và 6 tin là quá ít để kết luận.

3. **`hard_eval` — 50 case khó.** Mình đoán LLM đạt 85–95% và nhóm khó nhất với LLM là prompt injection. Thực tế LLM đạt 100% ở mọi nhóm, kể cả prompt injection, và prompt phòng thủ (4d) không cải thiện gì vì đã ở mức tối đa. **Vì sao lệch:** prompt zero-shot đã bọc tin nhắn trong `"""`, và LLM đã biết các chiêu lừa phổ biến từ quá trình pre-train. Chỉ khi mình tự viết case "nhắn nhầm số" thì LLM mới bị lừa (2/6 case).

## 4. Phân tích 50 case khó

| Model | F1 test thường (SUB) | F1 50 case khó | Mức tụt |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.824 | 0.378 | −0.45 |
| 2. BoW + MLP | 0.919 | 0.462 | **−0.46 (tụt nhiều nhất)** |
| 3a. Emb + kNN | 0.974 | 0.750 | −0.22 |
| 3b. Emb + LogReg | 1.000 | 0.600 | −0.40 |
| 3c. Emb + MLP head | 1.000 | 0.692 | −0.31 |
| 4a / 4d. LLM nhỏ | 1.000 | 1.000 | 0 |

**BoW + MLP tụt nhiều nhất**, nặng nhất ở nhóm **né từ khoá (0%)**, rồi đến spam giọng tự nhiên (10%). Embedding yếu ở nhóm **nói về spam (50%)** và **thao túng / lừa đảo** (38% với LogReg). Điểm trên test thường quá lạc quan vì test cùng phân phối với train (cùng bộ SMS năm 2012, cùng kiểu spam trúng thưởng và nhạc chuông).

**Case 1 — #29 (né từ khoá):** *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"*. BoW + MLP đoán sai (nhóm này đạt 0%). **Cơ chế:** BoW nhìn văn bản như các token rời rạc. `c0ngr4tulat10ns`, `w0n`, `cl4im` không có trong vocab 2000 từ nên bị bỏ qua hoàn toàn, và những từ còn lại (`you`, `a`, `at`) không đủ để đẩy lên spam. Embedding thì xử lý tốt nhóm này (83–100%), vì tokenizer tách từ thành các mảnh nhỏ nên vẫn nhận ra nghĩa "congratulations, won, claim".

**Case 2 — #100 (case tự viết, thao túng):** *"Hi, is this Linda? We were supposed to meet for coffee tomorrow"*. **Tất cả model đều sai, kể cả LLM.** **Cơ chế:** đây là tin mở màn của kiểu lừa "nhắn nhầm số": không có link, không có tiền, không có từ khoá, giọng hoàn toàn bình thường. Thông tin cần để nhận ra lừa đảo **không nằm trong tin nhắn**, ý đồ chỉ lộ ra ở các tin sau. Muốn phát hiện phải dùng thêm ngữ cảnh: lịch sử hội thoại, người gửi có trong danh bạ không.

## 5. Thử thách cá nhân

**Đề #2 (ML):** giảm vocab xuống 100 từ và tăng lên 5000 từ, train lại, báo F1 test và giải thích xu hướng.

**Cách làm:** dùng lại `build_vocab(texts_tr, max_size=...)`, `vectorize`, `train_logreg`, giữ nguyên lr = 0.5, 300 epochs, ngưỡng 0.5 như thí nghiệm 1.A để chỉ có kích thước vocab thay đổi. Đo thêm F1 train và % từ trong tin test không có trong vocab.

| vocab | F1 train | F1 test | precision | recall | % từ ngoài vocab (test) | train (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.840 | **0.770** | 1.0 | 0.626 | 52.0% | 0.5 |
| 2000 | 0.907 | **0.874** | 1.0 | 0.776 | 13.2% | 6.1 |
| 5000 | 0.914 | **0.868** | 1.0 | 0.766 | 9.3% | 15.3 |

**Giải thích:** tăng vocab giúp nhiều lúc đầu, rồi gần như không giúp nữa (lợi ích giảm dần). Với 100 từ, 52% số từ trong tin test bị bỏ qua, nhiều từ đặc trưng của spam (`claim`, `prize`, `150p`) bị mất, nên model thiếu thông tin (underfit): F1 train cũng thấp, recall chỉ 0.626. Từ 2000 lên 5000, F1 test gần như không đổi (chênh 1 tin spam), vì 3000 từ thêm vào đều là từ hiếm (số điện thoại, mã riêng): gradient nhỏ nên trọng số gần như bằng 0, và chúng hầu như không lặp lại ở test. Trong khi đó thời gian train tăng gấp 2.5 lần. **2000 từ là điểm cân bằng tốt nhất.** Muốn tăng recall nên cải thiện cách biểu diễn (chuẩn hoá văn bản, feature số điện thoại, embedding) thay vì chỉ tăng số từ.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây), p95 < 50 ms/tin.

- **Đạt độ trễ:** BoW + LogReg (p95 0.11 ms), BoW + MLP (0.62 ms), Encoder + LogReg 3b (37.7 ms). Không đạt: 3a (66 ms), 3c (59 ms), mọi LLM (1000–1500 ms).
- **Đạt cả độ trễ lẫn chất lượng:** **3b**, F1 test 0.967 (cao nhất trên toàn bộ test), F1 SUB 1.000.
- **Chi phí nếu dùng LLM nhỏ cho mọi tin:** khoảng 0.0075 USD / 1000 tin × 10 triệu tin ≈ **75 USD/ngày ≈ 2,300 USD/tháng**. LLM lớn khoảng 22,500 USD/tháng. Đó là chưa kể vi phạm SLA khoảng 20–30 lần.

**Phương án chọn: hệ thống lai 3 tầng.**
1. **Tầng 0 (luật):** chuẩn hoá văn bản (né từ khoá) và gắn cờ các cụm kiểu lệnh ("ignore previous instructions", "[SYSTEM").
2. **Tầng 1 (3b, Encoder + LogReg):** chạy cho mọi tin. p < 0.3 thì cho qua, p > 0.7 thì chặn.
3. **Tầng 2 (LLM nhỏ):** chỉ xử lý tin lưỡng lự (0.3 ≤ p ≤ 0.7) và tin bị tầng 0 gắn cờ, chạy bất đồng bộ.

**Đánh đổi chấp nhận:**
- 3b trên CPU Colab chỉ xử lý khoảng 30 tin/giây, nên cần GPU hoặc nhiều bản song song để đạt 116 tin/giây (giờ cao điểm còn hơn). p95 khoảng 38 ms cũng khá sát giới hạn. Phương án dự phòng là BoW + MLP (rất nhanh, F1 0.941).
- Để giữ p95 < 50 ms, tỉ lệ tin chuyển lên LLM phải dưới 5%, nếu không phải xử lý bất đồng bộ (gắn cờ, kiểm tra sau).
- Đổi lại, chi phí LLM giảm xuống cỡ vài chục đến 100 USD/tháng, mà vẫn tận dụng được điểm mạnh của LLM ở ca khó (trên 50 case khó: 3b đạt 60%, LLM đạt 100%).
- Ngưỡng 0.3 / 0.7 phải được chọn trên tập val.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Mình đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 4 · Chọn provider | Cách chọn nhà cung cấp LLM, thêm key vào Colab Secrets | Hướng dẫn thêm `OPENAI_API_KEY`, bật Notebook access | Chạy thử "Say hello in Vietnamese", nhận được câu trả lời (5.49 s) |
| 4.1 `call_openai` | Nhờ viết hàm | Hàm gọi `chat.completions.create` với `**provider_kwargs(max_tokens)`, đo latency | Test báo lỗi vì `.strip()` cắt mất khoảng trắng; bỏ `.strip()`, test pass |
| 4.2–4.3 prompt / `parse_label` | Nhờ viết prompt và hàm đọc nhãn | Prompt bọc tin trong `"""`; `parse_label` xử lý "not spam" trước, tách từ nguyên vẹn | `_run_check` pass; tự thử thêm "not spam", "I am not sure" |
| 4.4 `llm_classify_batch` | Nhờ viết hàm phân loại cả loạt + chi phí | Vòng lặp `retry_with_backoff(call_llm)`, đếm `n_invalid`, ước tính token | Test giả lập pass; thử 3 tin trước khi chạy cả SUB |
| 4.A debug | Hỏi vì sao F1 = 0 | Chỉ ra `invalid=40` là lỗi kỹ thuật, nghi `max_tokens = 3` quá nhỏ | Đặt `CLASSIFY_MAX_TOKENS = 20`, chạy lại → invalid = 0, F1 = 1.000 |
| 4.A, 4.B · dự đoán | Nhờ suy luận và viết câu `predict` | Lập luận về F1, tỉ lệ giá 10 lần, chi phí few-shot 2.5–3 lần | Đối chiếu kết quả: đúng tỉ lệ giá ~10 lần, few-shot ~2.5 lần |
| 4 · Câu hỏi 4 + Bắt lỗi 3 | Nhờ trả lời | Soạn câu trả lời về temperature, max_tokens, reasoning_effort, `parse_label_v2` | Đối chiếu với sự cố invalid = 40 |

**Điều mình hiểu ra hôm nay:**
- **Điểm cao trên tập test cùng phân phối với train không chứng minh model hiểu bài toán.** Encoder + LogReg đạt F1 1.000 trên SUB nhưng chỉ còn 0.60 trên case khó. Phải có bộ test khó hoặc dữ liệu mới mới biết model bền đến đâu.
- **Phải kiểm tra pipeline trước khi tin vào một con số.** F1 = 0 của LLM suýt khiến mình kết luận sai, trong khi thực ra là lỗi `max_tokens`. Từ nay mình luôn xem số câu trả lời không hợp lệ và đọc các ca sai.
- **Model mới hơn không có nghĩa là tốt hơn cho mọi bài.** Thí nghiệm bỏ ReLU cho thấy lợi thế của MLP trên BoW đến từ tối ưu tốt hơn chứ không phải từ lớp ẩn. Chọn model phải cân nhắc chất lượng, độ trễ, chi phí và dữ liệu có nhãn, không chỉ F1.