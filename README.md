# Facial Emotion Recognition Using AffectNet and MobileNetV2

Project này xây dựng pipeline nhận diện cảm xúc khuôn mặt từ ảnh tĩnh. 
## 1. Tổng quan

Mục tiêu hiện tại của project là train một baseline model phân loại cảm xúc khuôn mặt thành 7 lớp:

- `angry`
- `disgust`
- `fear`
- `happy`
- `neutral`
- `sad`
- `surprise`

Notebook chính sử dụng AffectNet từ Kaggle và MobileNetV2 pretrained trên ImageNet. MobileNetV2 là một kiến trúc CNN nhẹ, được dùng theo hướng transfer learning thay vì train một custom CNN từ đầu.

Kết quả hiện tại:

- Dataset hợp lệ sau indexing: `27,755` ảnh
- Train samples: `19,428`
- Validation samples: `4,163`
- Test samples: `4,164`
- Test accuracy: khoảng `61.74%`
- Model đã lưu: `artifacts/models/face_emotion_mobilenetv2_v2.keras`
- Metadata đã lưu: `artifacts/models/metadata_v2.json`

## 2. Cấu trúc project

```text
Emotion Detect/
├── artifacts/
│   └── models/
│       ├── face_emotion_mobilenetv2_v2.keras
│       └── metadata_v2.json
├── data/
├── docs/
│   ├── notebook_v2_review.md
│   ├── request.txt
│   └── report/
│       └── emotion_detection_report_overleaf.tex
├── notebooks/
│   ├── Face_Emotion_Detect_.ipynb
│   ├── NOTEBOOK_README.md
│   ├── dpl.pdf
│   ├── face-emotion-recognition-editable.pptx
│   └── r2.txt
├── outputs/
├── app/
└── README.md
```

Ghi chú:

- `app/` hiện chưa có source code chính thức. Phần app hoặc interface sẽ được bổ sung sau.
- `data/` là nơi dự kiến chứa dữ liệu raw/processed nếu chạy notebook local.
- `artifacts/models/` đang chứa model và metadata đã train.

## 3. Notebook chính

Notebook chính của project là:

```text
notebooks/Face_Emotion_Detect_.ipynb
```

Notebook này thực hiện toàn bộ pipeline training:

```text
Cài thư viện
-> import thư viện và tạo đường dẫn project
-> cấu hình dataset, class, hyperparameter
-> tìm hoặc tải AffectNet
-> chuẩn hóa label
-> tạo dataframe ảnh
-> preview và visualize dữ liệu
-> chia train/validation/test
-> tạo TensorFlow Dataset
-> build MobileNetV2 model
-> train classification head
-> fine-tune backbone
-> evaluate model
-> lưu model và metadata
-> demo predict ảnh đơn / detect face
```

File giải thích chi tiết từng cell của notebook:

```text
notebooks/NOTEBOOK_README.md
```

## 4. Dataset

Dataset được dùng là AffectNet từ Kaggle:

```text
mstjebashazida/affectnet
```

Notebook hỗ trợ nhiều cách lấy dữ liệu:

- Dùng dataset local khai báo trong `LOCAL_DATASET_ROOTS`
- Dùng dataset đã có trong `data/raw`
- Dùng Kaggle Input nếu chạy trên Kaggle Notebook
- Tải dataset bằng `kagglehub`

Sau khi index dữ liệu, notebook tạo dataframe `df_images`.

Các cột chính:

- `path`: đường dẫn ảnh
- `label`: nhãn cảm xúc dạng text
- `source`: nguồn dữ liệu
- `annotation_file`: file annotation nếu có
- `label_id`: ID số của label

## 5. Label normalization

Notebook chuẩn hóa label AffectNet về 7 class mục tiêu.

Mapping chính:

| AffectNet ID | Label |
|---:|---|
| 0 | `neutral` |
| 1 | `happy` |
| 2 | `sad` |
| 3 | `surprise` |
| 4 | `fear` |
| 5 | `disgust` |
| 6 | `angry` |
| 7-10 | ignored |

Các label ngoài phạm vi như `contempt`, `uncertain`, `none`, `non_face` bị loại bỏ.

Notebook cũng xử lý một số alias:

- `anger` -> `angry`
- `happiness` -> `happy`
- `sadness` -> `sad`
- `surprised` -> `surprise`

## 6. Phân tích và visualize dữ liệu

Notebook có phần `Data Preview` để kiểm tra dữ liệu trước khi train.

Các thông tin đã có:

- Tổng số ảnh hợp lệ
- Số nguồn dữ liệu
- Số file bị thiếu
- Bảng số lượng ảnh theo từng class
- Phần trăm từng class
- Biểu đồ phân bố class
- Biểu đồ nguồn dữ liệu
- Grid ảnh mẫu theo từng class

Phân bố class hiện tại:

| Class | Count |
|---|---:|
| `angry` | 3,218 |
| `disgust` | 2,477 |
| `fear` | 3,176 |
| `happy` | 5,044 |
| `neutral` | 5,126 |
| `sad` | 4,675 |
| `surprise` | 4,039 |

## 7. Train / validation / test split

Notebook chia dữ liệu theo tỉ lệ:

- Train: `70%`
- Validation: `15%`
- Test: `15%`

Việc chia dữ liệu dùng `train_test_split` với `stratify`, giúp giữ phân bố class tương đối ổn định giữa các tập.

Kết quả split hiện tại:

| Split | Samples |
|---|---:|
| Train | 19,428 |
| Validation | 4,163 |
| Test | 4,164 |

Các file split được lưu vào:

```text
data/processed/train_split.csv
data/processed/valid_split.csv
data/processed/test_split.csv
```

## 8. TensorFlow data pipeline

Notebook chuyển dataframe thành `tf.data.Dataset`.

Hàm `load_image` thực hiện:

- Đọc ảnh từ path
- Decode ảnh thành RGB
- Resize về `224 x 224`
- Cast sang `float32`

Hàm `make_dataset` thực hiện:

- Tạo dataset từ `path` và `label_id`
- Shuffle tập train
- Batch với `BATCH_SIZE = 32`
- Prefetch bằng `AUTOTUNE`

Notebook cũng tính `class_weight` bằng `compute_class_weight` để giảm ảnh hưởng của mất cân bằng class.

Class weight hiện tại:

| Class | Weight |
|---|---:|
| `angry` | 1.2319 |
| `disgust` | 1.6006 |
| `fear` | 1.2485 |
| `happy` | 0.7860 |
| `neutral` | 0.7735 |
| `sad` | 0.8482 |
| `surprise` | 0.9818 |

## 9. Model architecture

Model sử dụng MobileNetV2 pretrained trên ImageNet.

Kiến trúc:

```text
Input 224 x 224 x 3
-> Data augmentation
-> MobileNetV2 preprocess_input
-> MobileNetV2 backbone, include_top=False
-> GlobalAveragePooling2D
-> Dropout(0.35)
-> Dense(7, activation='softmax')
```

Data augmentation gồm:

- Random horizontal flip
- Random rotation `0.08`
- Random zoom `0.10`
- Random contrast `0.10`

Output của model là vector xác suất 7 chiều, tương ứng với 7 class cảm xúc.

## 10. Training strategy

Notebook train model theo 2 giai đoạn.

Giai đoạn 1: train classification head

- Freeze MobileNetV2 backbone
- Chỉ train phần head mới thêm
- Epochs: `8`
- Learning rate: `1e-3`
- Optimizer: Adam
- Loss: `sparse_categorical_crossentropy`
- Metric: `accuracy`

Giai đoạn 2: fine-tune backbone

- Unfreeze MobileNetV2
- Giữ frozen khoảng `70%` layer đầu
- Fine-tune khoảng `30%` layer cuối
- Epochs: `8`
- Learning rate: `1e-5`

Callbacks:

- `ModelCheckpoint`: lưu model tốt nhất theo `val_accuracy`
- `EarlyStopping`: dừng sớm nếu validation accuracy không cải thiện
- `ReduceLROnPlateau`: giảm learning rate nếu validation loss không cải thiện

Checkpoint tốt nhất được lưu tại:

```text
artifacts/models/best_emotion_model.keras
```

## 11. Evaluation

Sau khi train, notebook load lại checkpoint tốt nhất và evaluate trên test set.

Các phần đánh giá hiện có:

- Test loss
- Test accuracy
- Classification report
- Confusion matrix
- Accuracy curve
- Loss curve

Kết quả chính:

```text
test_accuracy = 0.6174351572990417
```

Tương đương:

```text
61.74%
```

Kết quả này là baseline ban đầu cho MobileNetV2 trên dữ liệu AffectNet đã xử lý.

## 12. Artifacts

Các artifact hiện có:

```text
artifacts/models/face_emotion_mobilenetv2_v2.keras
artifacts/models/metadata_v2.json
```

`face_emotion_mobilenetv2_v2.keras` là model đã train và lưu lại.

`metadata_v2.json` chứa:

- `class_names`
- `img_size`
- `kaggle_datasets`
- `dataset_roots`
- `train_samples`
- `valid_samples`
- `test_samples`
- `test_accuracy`
- `class_weights`

Metadata hiện tại xác nhận:

```json
{
  "img_size": [224, 224],
  "train_samples": 19428,
  "valid_samples": 4163,
  "test_samples": 4164,
  "test_accuracy": 0.6174351572990417
}
```

## 13. Report và slide

Các tài liệu hiện có:

```text
docs/report/emotion_detection_report_overleaf.tex
notebooks/dpl.pdf
notebooks/face-emotion-recognition-editable.pptx
```

Report LaTeX mô tả:

- Related work
- Dataset và preprocessing
- Model architecture
- Training/fine-tuning/evaluation
- Initial results
- Limitations
- Future work

PDF `dpl.pdf` là bản report ngắn hơn ở dạng paper.

PowerPoint `face-emotion-recognition-editable.pptx` là slide trình bày project.

## 14. Phần app/interface

Thư mục `app/` hiện chưa có source code chính thức.

Phần này sẽ được bổ sung sau. Dự kiến app/interface sẽ sử dụng:

```text
artifacts/models/face_emotion_mobilenetv2_v2.keras
artifacts/models/metadata_v2.json
```

Luồng app dự kiến:

```text
Upload/input image
-> load model
-> load metadata
-> resize image to 224 x 224
-> predict probabilities
-> return emotion label, confidence, probability table
```

Nếu cần detect nhiều khuôn mặt trong một ảnh, app có thể thêm face detector trước bước predict.

## 15. Hạn chế hiện tại

Những điểm project hiện chưa hoàn thiện:

- Chưa có app/interface chính thức.
- Chưa lưu classification report, confusion matrix và learning curves thành file riêng trong `artifacts/outputs`.
- Chưa có bước scan toàn dataset để phát hiện ảnh hỏng.
- Chưa kiểm tra duplicate ảnh bằng image hash.
- Chưa kiểm tra ảnh không có khuôn mặt.
- Chưa có so sánh với custom CNN, EfficientNet, ResNet hoặc ConvNeXt.
- Dependency chưa được pin version trong `requirements.txt`.

## 16. Hướng phát triển tiếp theo

Các việc nên bổ sung sau:

- Tạo `requirements.txt`.
- Lưu evaluation artifacts ra file:
  - classification report CSV/JSON
  - confusion matrix PNG/CSV
  - learning curves PNG
- Thêm data quality checks:
  - corrupted images
  - duplicate images
  - low-resolution images
  - non-face images
- Thêm model comparison:
  - custom CNN
  - EfficientNetV2
  - ResNet50
  - ConvNeXtTiny
- Xây dựng app/interface inference.
- Tách inference logic thành module riêng để app dùng lại.

## 17. Tóm tắt

Project hiện đã có một pipeline training hoàn chỉnh cho facial emotion recognition:

- Dataset: AffectNet
- Model: MobileNetV2 transfer learning
- Classes: 7 emotion classes
- Training: 2-stage training và fine-tuning
- Evaluation: accuracy, classification report, confusion matrix
- Result: khoảng `61.74%` test accuracy
- Artifact: model `.keras` và metadata `.json`

Phần implementation notebook và report đã có nền tảng tốt. Phần app/interface sẽ được phát triển ở bước tiếp theo.
