# Fine-tuning Mô hình Whisper

Dự án này cung cấp một công cụ để fine-tune mô hình Whisper của OpenAI trên tập dữ liệu âm thanh tùy chỉnh, hỗ trợ xử lý tiếng Việt (hoặc các ngôn ngữ khác). Dự án được tổ chức thành các module riêng biệt để dễ dàng bảo trì và tùy chỉnh.

## Cấu trúc dự án

```
src/
├── main.py                 # Điểm vào chính, xử lý tham số và điều phối huấn luyện
├── dataset.py              # Tải và tiền xử lý dữ liệu
├── model.py                # Khởi tạo mô hình, tokenizer và processor
├── trainer.py              # Thiết lập trainer và callback
├── callbacks.py            # Callback tùy chỉnh để hiển thị dự đoán mẫu
└── requirements.txt        # Danh sách thư viện yêu cầu
```

## Yêu cầu

-   Python 3.10+
-   Các thư viện được liệt kê trong `requirements.txt`

## Cài đặt

1. Tạo thư mục dự án và lưu các file theo cấu trúc trên.
2. Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

## Cách sử dụng

1.  **Chuẩn bị dữ liệu**:

    -   Tạo hai file CSV (`train.csv` và `eval.csv`) chứa cột `audio` (đường dẫn tương đối đến file âm thanh) và `transcription` (nội dung phiên âm).
    -   Đặt các file âm thanh trong một thư mục (ví dụ: `/path/to/audio`).

2.  **Chạy huấn luyện**:

    -   Sử dụng lệnh sau để chạy script với các tham số tùy chỉnh:

            python main.py \
                --train_metadata_csv /path/to/train.csv \
                --eval_metadata_csv /path/to/eval.csv \
                --audio_dir /path/to/audio \
                --output_dir ./whisper-finetuned \
                --model_name openai/whisper-small \
                --language vi \
                --task transcribe \
                --batch_size 16 \
                --gradient_accumulation_steps 1 \
                --learning_rate 1e-4 \
                --warmup_steps 50 \
                --num_train_epochs 3 \
                --logging_steps 20 \
                --save_steps 1000 \
                --eval_strategy steps \
                --eval_steps 500 \
                --generation_max_length 200 \
                --num_beams 4 \
                --enable_sample_predictions \
                --num_sample_predictions 5 \
                --report_to tensorboard \
                --seed 42

    -   Sử dụng lệnh sau để chạy script với các tham số (ví dụ):

              # mẫu chạy thử
              !python src/main.py \
                  --train_metadata_csv dataset/path/train.csv \
                  --eval_metadata_csv dataset/path/test.csv \
                  --audio_dir dataset/path/audio \
                  --output_dir ./whisper-finetuned \
                  --model_name openai/whisper-small \
                  --language vi \
                  --task transcribe \
                  --batch_size 8 \
                  --gradient_accumulation_steps 2 \
                  --learning_rate 1e-4 \
                  --warmup_steps 50 \
                  --num_train_epochs 3 \
                  --logging_steps 20 \
                  --save_steps 50 \
                  --save_strategy steps \
                  --eval_strategy steps \
                  --eval_steps 50 \
                  --generation_max_length 200 \
                  --load_best_model_at_end \
                  --num_beams 4 \
                  --enable_sample_predictions \
                  --num_sample_predictions 5 \
                  --report_to tensorboard \
                  --seed 42

---

3.  **Kết quả**:
    -   Mô hình và tokenizer được lưu trong thư mục `--output_dir`.
    -   Logs huấn luyện được ghi vào TensorBoard (nếu bật `--report_to tensorboard`).
    -   Dự đoán mẫu được hiển thị sau mỗi lần đánh giá nếu bật `--enable_sample_predictions`.

## Tham số tùy chỉnh

Dưới đây là các tham số có thể cấu hình qua dòng lệnh:

### Tham số dữ liệu

-   `--train_metadata_csv`: Đường dẫn đến file CSV chứa dữ liệu huấn luyện (bắt buộc).
-   `--eval_metadata_csv`: Đường dẫn đến file CSV chứa dữ liệu đánh giá (bắt buộc).
-   `--audio_dir`: Thư mục chứa file âm thanh (bắt buộc).
-   `--dataset_name`: Tên dataset (mặc định: `custom_dataset`).

### Tham số mô hình

-   `--model_name`: Mô hình Whisper tiền huấn luyện (mặc định: `openai/whisper-small`).
-   `--language`: Ngôn ngữ (mặc định: `vi`).
-   `--task`: Nhiệm vụ (mặc định: `transcribe`).

### Tham số huấn luyện

-   `--output_dir`: Thư mục lưu mô hình và logs (mặc định: `./whisper-small-vi`).
-   `--batch_size`: Kích thước batch trên mỗi thiết bị (mặc định: 8).
-   `--gradient_accumulation_steps`: Số bước tích lũy gradient (mặc định: 2).
-   `--learning_rate`: Tốc độ học (mặc định: 2e-4).
-   `--warmup_steps`: Số bước warmup (mặc định: 100).
-   `--num_train_epochs`: Số epoch huấn luyện (mặc định: 5).
-   `--max_steps`: Số bước huấn luyện tối đa (mặc định: -1, dùng epoch).
-   `--logging_steps`: Tần suất ghi log (mặc định: 10).
-   `--save_steps`: Tần suất lưu checkpoint (mặc định: 500).
-   `--eval_steps`: Tần suất đánh giá (mặc định: None, dùng `--eval_strategy`).
-   `--eval_strategy`: Chiến lược đánh giá (`no`, `steps`, `epoch`; mặc định: `epoch`).
-   `--save_strategy`: Chiến lược lưu mô hình (`no`, `steps`, `epoch`; mặc định: `epoch`).
-   `--logging_strategy`: Chiến lược ghi log (`no`, `steps`, `epoch`; mặc định: `steps`).
-   `--load_best_model_at_end`: Tải mô hình tốt nhất khi kết thúc (mặc định: True).
-   `--metric_for_best_model`: Metric chọn mô hình tốt nhất (mặc định: `wer`).
-   `--greater_is_better`: Metric càng cao càng tốt (mặc định: False, vì WER thấp hơn là tốt hơn).
-   `--fp16`: Sử dụng FP16 nếu GPU hỗ trợ (mặc định: auto).
-   `--gradient_checkpointing`: Bật gradient checkpointing để tiết kiệm bộ nhớ (mặc định: True).

### Tham số sinh câu

-   `--generation_max_length`: Độ dài tối đa của chuỗi sinh ra (mặc định: 225).
-   `--num_beams`: Số beam cho beam search (mặc định: 2).
-   `--predict_with_generate`: Sử dụng generate() khi đánh giá (mặc định: True).

### Tham số callback

-   `--enable_sample_predictions`: Bật callback hiển thị dự đoán mẫu (mặc định: True).
-   `--num_sample_predictions`: Số mẫu dự đoán hiển thị (mặc định: 3).

### Tham số khác

-   `--push_to_hub`: Đẩy mô hình lên Hugging Face Hub (mặc định: False).
-   `--report_to`: Công cụ ghi log (mặc định: `tensorboard`).
-   `--seed`: Hạt giống ngẫu nhiên (mặc định: 42).

## Lưu ý

-   File CSV phải chứa cột `audio` (đường dẫn tương đối đến file âm thanh) và `transcription` (nội dung phiên âm).
-   Đảm bảo file âm thanh có thể truy cập trong thư mục `--audio_dir`.
-   Nếu gặp lỗi bộ nhớ, giảm `--batch_size` hoặc tăng `--gradient_accumulation_steps`.
-   Để tắt callback dự đoán mẫu, sử dụng `--no-enable_sample_predictions`.

## Tác giả

Dự án được phát triển để thực hiện fine-tuning mô hình Whisper với dữ liệu tùy chỉnh.

**github: _nguyendevwk_**
