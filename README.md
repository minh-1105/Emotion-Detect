# Emotion Detect

Tài liệu này giải thích chi tiết notebook mới:

- [Face_Emotion_Detect_Multi_v2.ipynb](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\notebooks\Face_Emotion_Detect_Multi_v2.ipynb)

Notebook này dùng để:

1. Tìm hoặc tải dataset AffectNet
2. Chuẩn hóa nhãn cảm xúc về 7 lớp
3. Tạo tập train, validation, test
4. Huấn luyện mô hình nhận diện cảm xúc bằng `MobileNetV2`
5. Đánh giá mô hình
6. Lưu model và metadata để dùng cho ứng dụng sau này

## 1. Mục tiêu của notebook

Notebook `Face_Emotion_Detect_Multi_v2.ipynb` là một pipeline huấn luyện hoàn chỉnh cho bài toán phân loại cảm xúc khuôn mặt.

Đầu vào:

- Dataset kiểu AffectNet
- Hoặc dataset có file CSV annotation
- Hoặc dataset dạng thư mục lớp, ví dụ `happy/*.jpg`, `sad/*.png`

Đầu ra:

- Model tốt nhất trong quá trình train
- Model cuối cùng để triển khai
- Metadata mô tả model
- Các file CSV chứa danh sách train/valid/test
- Báo cáo đánh giá và confusion matrix

## 2. Cấu trúc thư mục mà notebook sử dụng

Notebook mới được viết để bám theo cấu trúc repo hiện tại:

```text
Emotion Detect/
├─ app/
├─ artifacts/
│  ├─ models/
│  └─ outputs/
├─ data/
│  ├─ processed/
│  └─ raw/
├─ docs/
└─ notebooks/
   ├─ Face_Emotion_Detect_Multi.ipynb
   └─ Face_Emotion_Detect_Multi_v2.ipynb
```

Ý nghĩa:

- `data/raw/`: nơi chứa dataset gốc nếu bạn tải sẵn về máy
- `data/processed/`: nơi notebook lưu các file split train/valid/test
- `artifacts/models/`: nơi lưu model `.keras` và file metadata
- `artifacts/outputs/`: nơi để các output phụ trợ nếu cần
- `notebooks/`: nơi chứa notebook huấn luyện

## 3. Tổng quan luồng hoạt động

Notebook chạy theo thứ tự này:

1. Cài thư viện
2. Import package và xác định đường dẫn dự án
3. Khai báo biến cấu hình
4. Tìm dataset local hoặc tải từ Kaggle
5. Quét dữ liệu ảnh và chuẩn hóa nhãn
6. Tạo train/validation/test split có stratify
7. Tạo `tf.data.Dataset`
8. Tính `class_weight`
9. Khởi tạo mô hình `MobileNetV2`
10. Train giai đoạn 1 với backbone đóng băng
11. Fine-tune giai đoạn 2 với một phần backbone được mở
12. Đánh giá mô hình trên test set
13. Lưu model và metadata
14. Chạy dự đoán trên ảnh đơn hoặc ảnh có nhiều khuôn mặt

## 4. Giải thích chi tiết từng phần

### 4.1. Cài thư viện

Notebook dùng:

- `kagglehub`: tải dataset từ Kaggle
- `tensorflow`: train model
- `opencv-python`: detect face
- `scikit-learn`: chia dữ liệu và tính metric
- `matplotlib`: vẽ biểu đồ
- `pandas`: xử lý bảng dữ liệu
- `pillow`: đọc ảnh

### 4.2. Xác định đường dẫn dự án

Notebook tự suy ra thư mục gốc dự án từ thư mục đang chạy.

Các biến đường dẫn:

- `NOTEBOOK_DIR`: thư mục hiện tại khi notebook chạy
- `PROJECT_ROOT`: thư mục gốc repo
- `DATA_DIR`: thư mục `data`
- `RAW_DIR`: `data/raw`
- `PROCESSED_DIR`: `data/processed`
- `ARTIFACTS_DIR`: `artifacts`
- `MODEL_DIR`: `artifacts/models`
- `OUTPUT_DIR`: `artifacts/outputs`
- `UPLOAD_DIR`: thư mục con để chứa ảnh test nếu cần

Notebook tự tạo các thư mục này nếu chưa có.

### 4.3. Cấu hình chính

Đây là các biến bạn cần hiểu rõ nhất.

#### Biến tái lập kết quả

- `SEED = 42`

Ý nghĩa:

- Đồng bộ random cho Python, NumPy, TensorFlow
- Giúp việc chia dữ liệu và train ổn định hơn giữa các lần chạy

#### Biến kích thước ảnh

- `IMG_SIZE = (224, 224)`

Ý nghĩa:

- Mọi ảnh được resize về `224x224`
- Đây là kích thước phù hợp với `MobileNetV2`

#### Biến batch

- `BATCH_SIZE = 32`

Ý nghĩa:

- Số ảnh đi qua model trong một batch
- Tăng batch sẽ nhanh hơn nếu GPU đủ RAM
- Nếu máy yếu hoặc thiếu VRAM thì giảm xuống `16` hoặc `8`

#### Biến số epoch

- `EPOCHS_HEAD = 8`
- `EPOCHS_FINE_TUNE = 8`

Ý nghĩa:

- Giai đoạn 1 train phần head classifier
- Giai đoạn 2 fine-tune thêm backbone

#### Tỷ lệ chia dữ liệu

- `VALID_SIZE = 0.15`
- `TEST_SIZE = 0.15`

Ý nghĩa:

- 15% cho validation
- 15% cho test
- 70% còn lại cho train

#### Learning rate

- `LEARNING_RATE_HEAD = 1e-3`
- `LEARNING_RATE_FINE = 1e-5`

Ý nghĩa:

- Head train nhanh hơn với LR lớn hơn
- Fine-tuning cần LR thấp để tránh phá hỏng trọng số pretrained

#### Mixed precision

- `USE_MIXED_PRECISION = True`

Ý nghĩa:

- Nếu có GPU phù hợp, TensorFlow dùng tính toán mixed precision
- Có thể tăng tốc train và giảm dùng VRAM

#### Điều khiển nguồn dữ liệu

- `COPY_DATASETS_TO_RAW = False`
- `KAGGLE_DATASETS = ['mstjebashazida/affectnet']`
- `LOCAL_DATASET_ROOTS = []`

Ý nghĩa:

- `LOCAL_DATASET_ROOTS`: nếu bạn có dataset local thì điền vào đây
- `KAGGLE_DATASETS`: slug dataset Kaggle để notebook dùng khi cần tải
- `COPY_DATASETS_TO_RAW`: nếu `True`, notebook sẽ copy dataset đã tải về vào `data/raw`

Khuyến nghị:

- Với dataset lớn, nên giữ `False`
- Nếu muốn repo có cấu trúc dữ liệu cục bộ rõ ràng thì có thể bật `True`, nhưng sẽ tốn dung lượng

### 4.4. Nhãn cảm xúc

Notebook chuẩn hóa về 7 lớp:

- `angry`
- `disgust`
- `fear`
- `happy`
- `neutral`
- `sad`
- `surprise`

Biến liên quan:

- `CLASS_NAMES`
- `CLASS_TO_ID`
- `AFFECTNET_ID_TO_LABEL`
- `LABEL_ALIASES`

Ý nghĩa:

- `CLASS_NAMES`: danh sách lớp chuẩn
- `CLASS_TO_ID`: map tên lớp sang số
- `AFFECTNET_ID_TO_LABEL`: map nhãn số AffectNet sang tên lớp
- `LABEL_ALIASES`: chuẩn hóa các cách ghi tên khác nhau

Các nhãn bị bỏ qua:

- `contempt`
- `none`
- `uncertain`
- `non-face`

Lý do:

- Notebook đang train theo bài toán 7-class

## 5. Cách notebook tìm dữ liệu

Notebook tìm dữ liệu theo thứ tự ưu tiên:

1. `LOCAL_DATASET_ROOTS` nếu bạn chỉ định thủ công
2. Các thư mục đã có sẵn trong `data/raw/`
3. `/kaggle/input/` nếu đang chạy trên Kaggle Notebook
4. `kagglehub.dataset_download(...)` nếu chưa có dữ liệu local

Điều này có nghĩa:

- Nếu `data/raw/` đã có dataset, notebook sẽ không cần tải thêm
- Nếu `data/raw/` rỗng, bạn cần Kaggle credential hoặc chạy trên Kaggle

## 6. Notebook đọc dữ liệu như thế nào

Notebook hỗ trợ 2 kiểu chính.

### 6.1. Dataset có CSV annotation

Notebook sẽ:

1. Quét toàn bộ file `.csv`
2. Tìm cột chứa đường dẫn ảnh
3. Tìm cột chứa nhãn cảm xúc
4. Đọc CSV theo từng `chunk`
5. Map nhãn về 7 lớp chuẩn
6. Tìm file ảnh thật trên đĩa

Các biến hỗ trợ:

- `AFFECTNET_LABEL_COLUMNS`
- `AFFECTNET_PATH_COLUMNS`
- `find_existing_image(...)`
- `collect_affectnet_csv_records(...)`

### 6.2. Dataset dạng thư mục lớp

Nếu không tìm thấy CSV hợp lệ, notebook sẽ:

1. Duyệt tất cả file ảnh
2. Suy ra lớp từ tên thư mục
3. Gán nhãn tương ứng

Hàm liên quan:

- `infer_label_from_path(...)`
- `collect_folder_label_records(...)`

### 6.3. Biến dữ liệu đầu ra

Sau khi quét xong, notebook tạo DataFrame:

- `df_images`

Các cột chính:

- `path`: đường dẫn tuyệt đối tới file ảnh
- `label`: nhãn text
- `source`: tên nguồn dataset
- `annotation_file`: tên file CSV nếu có
- `label_id`: mã số lớp

## 7. Chia train / validation / test

Notebook dùng `train_test_split` với `stratify`.

Ý nghĩa:

- Tỷ lệ từng lớp sẽ được giữ tương đối đồng đều giữa train/valid/test
- Điều này quan trọng với bài toán cảm xúc vì class thường lệch mạnh

Biến tạo ra:

- `train_df`
- `valid_df`
- `test_df`

Notebook cũng lưu:

- [train_split.csv](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\data\processed\train_split.csv)
- [valid_split.csv](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\data\processed\valid_split.csv)
- [test_split.csv](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\data\processed\test_split.csv)

Lưu ý:

- Các file này chỉ xuất hiện sau khi chạy notebook

## 8. TensorFlow Dataset pipeline

Notebook chuyển DataFrame thành `tf.data.Dataset`.

Các bước:

1. Đọc file ảnh từ `path`
2. Decode ảnh RGB
3. Resize về `224x224`
4. Chuyển sang `float32`
5. Batch
6. Prefetch

Hàm chính:

- `load_image(path, label)`
- `make_dataset(dataframe, training=False)`

Biến đầu ra:

- `train_ds`
- `valid_ds`
- `test_ds`

## 9. Class weight là gì và vì sao cần

Notebook tính:

- `class_weights`

Bằng:

- `compute_class_weight(..., class_weight='balanced', ...)`

Ý nghĩa:

- Nếu lớp nào ít ảnh hơn, loss của lớp đó sẽ được tăng trọng số
- Giúp model đỡ thiên về các lớp xuất hiện nhiều như `happy` hoặc `neutral`

Điều này đặc biệt quan trọng với AffectNet vì dữ liệu thực tế thường không cân bằng.

## 10. Kiến trúc model

Model chính là `MobileNetV2` pretrained trên ImageNet.

### 10.1. Backbone

```python
base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=IMG_SIZE + (3,),
)
```

Ý nghĩa:

- Dùng mô hình pretrained để tận dụng đặc trưng thị giác đã học sẵn
- `include_top=False` để bỏ phần classifier gốc của ImageNet

### 10.2. Data augmentation

Notebook thêm:

- lật ngang
- xoay nhẹ
- zoom nhẹ
- đổi tương phản nhẹ

Mục tiêu:

- Giảm overfitting
- Tăng đa dạng dữ liệu train

### 10.3. Head classifier

Luồng cuối của model:

1. `preprocess_input`
2. `base_model`
3. `GlobalAveragePooling2D`
4. `Dropout(0.35)`
5. `Dense(7, softmax)`

Ý nghĩa từng phần:

- `GlobalAveragePooling2D`: gom feature map thành vector
- `Dropout(0.35)`: giảm overfitting
- `Dense(..., softmax)`: xuất xác suất cho 7 lớp

### 10.4. Output của model

Model trả về vector xác suất 7 chiều:

```text
[P(angry), P(disgust), P(fear), P(happy), P(neutral), P(sad), P(surprise)]
```

Lớp dự đoán cuối cùng là phần tử có xác suất lớn nhất.

## 11. Hai giai đoạn train

### 11.1. Giai đoạn 1: train head

Ở giai đoạn này:

- `base_model.trainable = False`

Ý nghĩa:

- Khóa toàn bộ backbone
- Chỉ train phần classifier phía trên

Mục tiêu:

- Để head học cách map đặc trưng sang 7 lớp cảm xúc
- Tránh làm hỏng trọng số pretrained ngay từ đầu

### 11.2. Giai đoạn 2: fine-tune

Ở giai đoạn này:

- `base_model.trainable = True`
- Chỉ mở phần 30% layer cuối để train tiếp

Mục tiêu:

- Tinh chỉnh backbone cho đặc thù ảnh khuôn mặt và cảm xúc

## 12. Callback trong quá trình train

Notebook dùng 3 callback quan trọng.

### 12.1. ModelCheckpoint

Lưu model tốt nhất theo:

- `val_accuracy`

File:

- [best_emotion_model.keras](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\artifacts\models\best_emotion_model.keras)

Ý nghĩa:

- Dù epoch cuối không tốt nhất, bạn vẫn giữ được model tốt nhất trên validation

### 12.2. EarlyStopping

Theo dõi:

- `val_accuracy`

Ý nghĩa:

- Nếu mô hình không cải thiện sau vài epoch, notebook dừng sớm
- Tránh train vô ích và giảm overfitting

### 12.3. ReduceLROnPlateau

Theo dõi:

- `val_loss`

Ý nghĩa:

- Khi learning bị chững, notebook tự giảm learning rate

## 13. Sau khi train xong, model sẽ như thế nào

Sau khi train xong, bạn sẽ có 2 mốc quan trọng:

### 13.1. Best model

File:

- [best_emotion_model.keras](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\artifacts\models\best_emotion_model.keras)

Đây là model có `val_accuracy` tốt nhất trong toàn bộ quá trình train.

### 13.2. Final deploy model

File:

- [face_emotion_mobilenetv2_v2.keras](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\artifacts\models\face_emotion_mobilenetv2_v2.keras)

Đây là bản model được lưu lại sau khi notebook load `best_model` và save sang tên ổn định hơn cho việc tích hợp ứng dụng.

### 13.3. Metadata

File:

- [metadata_v2.json](C:\Users\minht\OneDrive\Tài liệu\Emotion Detect\artifacts\models\metadata_v2.json)

Nội dung metadata gồm:

- `class_names`
- `img_size`
- `kaggle_datasets`
- `dataset_roots`
- `train_samples`
- `valid_samples`
- `test_samples`
- `test_accuracy`
- `class_weights`

Ý nghĩa:

- Đây là file rất quan trọng cho app inference
- App có thể đọc file này để biết model dùng kích thước ảnh nào và thứ tự class là gì

## 14. Các chỉ số đánh giá sẽ xuất hiện

Notebook sinh ra các chỉ số sau.

### 14.1. `accuracy`

Xuất hiện trong train:

- `accuracy`
- `val_accuracy`

Ý nghĩa:

- `accuracy`: độ chính xác trên tập train
- `val_accuracy`: độ chính xác trên tập validation

Nếu:

- `accuracy` tăng mạnh nhưng `val_accuracy` kém hoặc đứng yên

thì thường là dấu hiệu overfitting.

### 14.2. `loss`

Xuất hiện trong train:

- `loss`
- `val_loss`

Ý nghĩa:

- Mức sai số của mô hình
- Thường càng thấp càng tốt

### 14.3. `test_loss`

Được tính sau khi load model tốt nhất và evaluate trên `test_ds`.

Ý nghĩa:

- Sai số cuối cùng trên test set

### 14.4. `test_accuracy`

Được tính sau khi evaluate trên test set.

Ý nghĩa:

- Đây là chỉ số tổng quan quan trọng nhất để xem model dùng được đến đâu trên dữ liệu chưa thấy

### 14.5. Classification report

Notebook dùng:

- `classification_report(y_true, y_pred, target_names=CLASS_NAMES)`

Report này thường gồm:

- `precision`
- `recall`
- `f1-score`
- `support`

Ý nghĩa:

- `precision`: model đoán lớp đó thì đúng bao nhiêu phần trăm
- `recall`: ảnh thật thuộc lớp đó thì model bắt được bao nhiêu phần trăm
- `f1-score`: cân bằng giữa precision và recall
- `support`: số mẫu thật của lớp đó trong test set

Đây là phần rất quan trọng vì bài toán cảm xúc thường không cân bằng. Chỉ nhìn `accuracy` là chưa đủ.

### 14.6. Confusion matrix

Notebook vẽ confusion matrix.

Ý nghĩa:

- Cho biết lớp nào hay bị nhầm sang lớp nào
- Ví dụ thường gặp:
  - `fear` nhầm sang `surprise`
  - `sad` nhầm sang `neutral`
  - `disgust` nhầm sang `angry`

Confusion matrix là nơi tốt nhất để hiểu mô hình đang sai kiểu gì.

## 15. Hiện tại chưa có chỉ số thực tế

Trong repo hiện tại:

- Chưa có dataset local trong `data/raw/`
- Chưa có model đã train trong `artifacts/models/`
- Chưa có file split trong `data/processed/`

Vì vậy:

- README này giải thích chính xác notebook sẽ tạo ra các chỉ số gì
- Nhưng chưa thể ghi số liệu thực tế như `test_accuracy = ...` cho đến khi notebook được chạy

## 16. Cách đọc kết quả sau khi chạy notebook

Sau khi chạy xong, bạn nên kiểm tra theo thứ tự:

1. `Images found`
2. Bảng phân phối số lượng mẫu theo lớp
3. Bảng split train/valid/test
4. `val_accuracy` tốt nhất trong lúc train
5. `test_accuracy`
6. `classification_report`
7. `confusion_matrix`
8. File model và metadata được lưu thành công hay chưa

## 17. Hàm suy luận sau khi train

Notebook có 2 hàm suy luận chính.

### 17.1. `predict_emotion(image_path, model=best_model)`

Chức năng:

- Nhận một ảnh
- Resize về `IMG_SIZE`
- Chạy model
- Trả về bảng xác suất theo từng cảm xúc

Kết quả:

- Ảnh được hiển thị
- Tiêu đề ảnh chứa lớp dự đoán và độ tin cậy
- Trả về DataFrame xác suất

### 17.2. `detect_faces_and_predict(image_path, model=best_model)`

Chức năng:

- Dùng Haar Cascade của OpenCV để tìm khuôn mặt
- Crop từng khuôn mặt
- Chạy dự đoán cảm xúc cho từng face
- Vẽ bounding box và nhãn lên ảnh

Lưu ý:

- Haar Cascade nhanh nhưng không phải detector tốt nhất
- Nếu cần chất lượng cao hơn, có thể thay bằng MTCNN hoặc MediaPipe

## 18. Các cải tiến đã được đưa vào notebook mới

So với notebook gốc, bản `v2` cải thiện các điểm sau:

1. Sửa lỗi biến `df` gây hỏng bước split
2. Bám theo cấu trúc thư mục mới của dự án
3. Ưu tiên dùng dữ liệu local trong `data/raw/`
4. Lưu split CSV để tiện tái sử dụng
5. Lưu metadata chi tiết hơn
6. Thêm `class_weight` để xử lý mất cân bằng lớp
7. Bật mixed precision khi có GPU

## 19. Các giới hạn hiện tại

Notebook hiện vẫn có các giới hạn sau:

1. Chưa có bước detect face trước khi train
2. Chưa cache sẵn ảnh đã resize ra file
3. Chưa có TTA
4. Chưa có export sang ONNX hoặc TensorFlow Lite
5. Chưa có logging chuẩn như TensorBoard hoặc MLflow
6. Detector face khi inference còn là Haar Cascade

## 20. Khi nào cần tối ưu thêm

Bạn nên cân nhắc tối ưu tiếp nếu gặp một trong các trường hợp:

1. `val_accuracy` thấp và không cải thiện
2. `test_accuracy` thấp hơn nhiều so với `val_accuracy`
3. Lớp `fear`, `disgust`, `sad` có `f1-score` thấp
4. Train quá chậm trên máy hiện tại
5. Ứng dụng cần chạy realtime

Các hướng tối ưu tiếp:

1. Chuyển sang `EfficientNetB0` hoặc `EfficientNetV2B0`
2. Dùng face crop trước khi train
3. Tăng augmentation hợp lý
4. Dùng focal loss nếu class imbalance nặng
5. Tối ưu inference model cho app

## 21. Cách chạy notebook an toàn

Khuyến nghị chạy theo thứ tự:

1. Mở notebook `Face_Emotion_Detect_Multi_v2.ipynb`
2. Chạy cell cài thư viện
3. Kiểm tra `LOCAL_DATASET_ROOTS` hoặc Kaggle credential
4. Chạy tới cell `dataset_roots` để xác nhận nguồn dữ liệu
5. Kiểm tra `Images found`
6. Kiểm tra phân phối lớp
7. Bắt đầu train
8. Sau khi train xong, xác nhận model đã được lưu vào `artifacts/models/`

## 22. Tóm tắt ngắn

Notebook mới là một pipeline train emotion classifier hoàn chỉnh với:

- Input linh hoạt từ Kaggle hoặc local
- Chuẩn hóa dữ liệu theo 7 lớp cảm xúc
- Transfer learning bằng `MobileNetV2`
- Huấn luyện 2 giai đoạn
- Đánh giá bằng accuracy, loss, classification report, confusion matrix
- Lưu model và metadata để dùng cho ứng dụng

Phần còn thiếu hiện tại không nằm ở logic notebook mà nằm ở dữ liệu thực tế:

- Bạn cần dataset local hoặc cần quyền tải từ Kaggle
- Khi notebook được chạy xong, mới có các chỉ số thực tế để báo cáo chính xác
