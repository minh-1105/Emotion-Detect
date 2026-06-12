# README chi tiết notebook `Face_Emotion_Detect_.ipynb`

File này giải thích luồng hoạt động của notebook `Face_Emotion_Detect_.ipynb`: từng cell làm gì, xử lý dữ liệu như thế nào, các biến/hàm quan trọng và kết quả được tạo ra.

## 1. Mục tiêu notebook

Notebook xây dựng một pipeline nhận diện cảm xúc khuôn mặt trên bộ dữ liệu AffectNet. Pipeline đi từ bước chuẩn bị dữ liệu đến train, fine-tune, evaluate và lưu model.

Luồng tổng quát:

```text
Cài thư viện
-> import thư viện và tạo đường dẫn dự án
-> cấu hình class, dataset, hyperparameter
-> tìm hoặc tải AffectNet
-> chuẩn hóa nhãn và tạo dataframe ảnh
-> preview/visualize dữ liệu
-> chia train/validation/test
-> tạo TensorFlow Dataset
-> build MobileNetV2 model
-> train head
-> fine-tune backbone
-> evaluate test set
-> lưu model và metadata
-> demo predict ảnh đơn / detect face
```

Model chính là MobileNetV2 pretrained trên ImageNet. Đây là một kiến trúc CNN nhẹ. Notebook không train một custom CNN từ đầu, mà dùng transfer learning với MobileNetV2 rồi thêm classification head cho 7 class cảm xúc.

## 2. Cell 1 - Giới thiệu notebook

Cell markdown này giới thiệu notebook `Face Emotion Detection - AffectNet Training (v2)`.

Nội dung chính:

- Notebook dùng AffectNet.
- Sửa lại xử lý đường dẫn dataset.
- Dùng thư mục ổn định trong project như `data/raw`, `data/processed`, `artifacts/models`.
- Thêm class weighting để xử lý mất cân bằng class.
- Lưu split CSV và metadata để dùng lại cho app hoặc inference.

Cell này không chạy code, chỉ mô tả mục tiêu và điểm cải tiến.

## 3. Cell 2 - Tiêu đề cài thư viện

Cell markdown `Install Libraries`.

Mục đích:

- Báo rằng cell tiếp theo dùng để cài package.
- Chỉ cần chạy một lần trong môi trường mới như Colab.

## 4. Cell 3 - Cài thư viện

```python
!pip install -q kagglehub tensorflow opencv-python scikit-learn matplotlib pandas pillow
```

Cell này cài các thư viện cần thiết:

- `kagglehub`: tải dataset từ Kaggle.
- `tensorflow`: build, train, evaluate model.
- `opencv-python`: đọc ảnh và detect face bằng Haar Cascade.
- `scikit-learn`: chia train/test, tính class weight, classification report, confusion matrix.
- `matplotlib`: vẽ biểu đồ và hiển thị ảnh.
- `pandas`: xử lý bảng dữ liệu ảnh.
- `pillow`: đọc và xử lý ảnh bằng `PIL.Image`.

Cell này đặc biệt hữu ích trên Google Colab. Nếu chạy local và đã cài sẵn thư viện thì có thể bỏ qua.

## 5. Cell 4 - Tiêu đề import và đường dẫn

Cell markdown `Imports and Project Paths`.

Nó báo rằng cell tiếp theo sẽ:

- Import thư viện.
- Xác định root folder của project.
- Tạo các folder cần thiết.

## 6. Cell 5 - Import thư viện và tạo cấu trúc thư mục

Cell này import toàn bộ thư viện chính và thiết lập các biến đường dẫn.

Các thư viện chính:

- `json`: đọc/ghi metadata.
- `os`: đọc biến môi trường, chỉnh permission Kaggle credential.
- `random`: cố định random seed.
- `Path` từ `pathlib`: quản lý đường dẫn.
- `cv2`: xử lý ảnh bằng OpenCV.
- `matplotlib.pyplot`: vẽ biểu đồ và ảnh.
- `numpy`: xử lý array.
- `pandas`: xử lý dataframe.
- `tensorflow`, `keras`, `layers`: train deep learning model.
- `PIL.Image`: đọc ảnh.
- `train_test_split`: chia train/validation/test.
- `compute_class_weight`: tính trọng số class.
- `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`: đánh giá model.
- `kagglehub`: tải dataset, nếu có cài.

Các biến đường dẫn:

- `NOTEBOOK_DIR`: thư mục hiện tại nơi notebook chạy.
- `PROJECT_ROOT`: thư mục gốc project. Nếu thư mục hiện tại có `data` thì lấy chính nó, nếu không lấy parent.
- `DATA_DIR`: `PROJECT_ROOT / 'data'`.
- `RAW_DIR`: nơi chứa dataset thô, `data/raw`.
- `PROCESSED_DIR`: nơi lưu split CSV, `data/processed`.
- `ARTIFACTS_DIR`: thư mục artifact, `artifacts`.
- `MODEL_DIR`: nơi lưu model và metadata, `artifacts/models`.
- `OUTPUT_DIR`: nơi lưu output khác, `artifacts/outputs`.
- `UPLOAD_DIR`: nơi lưu ảnh upload, `artifacts/outputs/uploaded_images`.

Cell tạo các folder trên bằng:

```python
path.mkdir(parents=True, exist_ok=True)
```

Sau đó in ra:

- TensorFlow version.
- GPU devices.
- Project root.
- Raw data dir.
- Model dir.

Ý nghĩa trong luồng hoạt động:

- Đây là cell nền tảng. Các cell sau đều dùng lại các đường dẫn này.
- Nếu `PROJECT_ROOT` xác định sai thì các file model, metadata, split CSV có thể bị lưu sai chỗ.

## 7. Cell 6 - Hàm hiển thị grid ảnh

Cell này định nghĩa helper function:

```python
show_image_grid(image_paths, titles=None, cols=4, figsize_per_img=3)
```

Mục đích:

- Hiển thị nhiều ảnh cùng lúc theo dạng lưới.
- Dùng để preview dữ liệu hoặc demo ảnh mẫu.

Tham số:

- `image_paths`: danh sách đường dẫn ảnh.
- `titles`: danh sách tiêu đề tương ứng từng ảnh, có thể là label.
- `cols`: số cột trong grid, mặc định 4.
- `figsize_per_img`: kích thước mỗi ảnh trong figure.

Luồng xử lý:

1. Chuyển `image_paths` thành list.
2. Nếu không có ảnh thì in `No images to display.` và dừng.
3. Tính số dòng bằng `ceil(len(image_paths) / cols)`.
4. Tạo figure và axes bằng `plt.subplots`.
5. Duyệt từng ô trong grid.
6. Với mỗi ảnh:
   - Mở ảnh bằng `Image.open`.
   - Convert sang RGB.
   - Hiển thị bằng `ax.imshow`.
   - Nếu có `titles`, đặt title cho ảnh.
7. Nếu ảnh lỗi, hiển thị thông báo `Cannot read`.

Cell này không tự chạy preview. Nó chỉ tạo hàm dùng lại.

## 8. Cell 7 - Tiêu đề cấu hình

Cell markdown `Configuration`.

Nó giải thích rằng nếu đã có AffectNet local thì có thể đặt đường dẫn vào `LOCAL_DATASET_ROOTS`. Nếu `data/raw` đã có dataset thì notebook sẽ dùng lại và không tải từ Kaggle.

## 9. Cell 8 - Cấu hình toàn bộ experiment

Cell này là nơi định nghĩa hyperparameter, class, mapping label và seed.

Các biến training:

- `SEED = 42`: cố định random để split và training ổn định hơn.
- `IMG_SIZE = (224, 224)`: resize ảnh về 224x224, phù hợp MobileNetV2.
- `BATCH_SIZE = 32`: số ảnh mỗi batch.
- `EPOCHS_HEAD = 8`: số epoch train classification head.
- `EPOCHS_FINE_TUNE = 8`: số epoch fine-tune backbone.
- `VALID_SIZE = 0.15`: validation chiếm 15%.
- `TEST_SIZE = 0.15`: test chiếm 15%.
- `LEARNING_RATE_HEAD = 1e-3`: learning rate khi train head.
- `LEARNING_RATE_FINE = 1e-5`: learning rate khi fine-tune.
- `USE_MIXED_PRECISION = True`: bật mixed precision nếu có GPU phù hợp.
- `COPY_DATASETS_TO_RAW = False`: có copy dataset tải về vào `data/raw` hay không.

Các biến dataset:

- `KAGGLE_DATASETS = ['mstjebashazida/affectnet']`: dataset AffectNet trên Kaggle.
- `LOCAL_DATASET_ROOTS = []`: nếu có dataset local thì thêm path vào đây.

Các biến class:

- `CLASS_NAMES`: 7 class cảm xúc:
  - `angry`
  - `disgust`
  - `fear`
  - `happy`
  - `neutral`
  - `sad`
  - `surprise`
- `CLASS_TO_ID`: map tên class sang số ID.
- `IMAGE_EXTS`: các đuôi file ảnh hợp lệ như `.jpg`, `.png`, `.webp`.

Mapping AffectNet:

```python
AFFECTNET_ID_TO_LABEL = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'surprise',
    4: 'fear',
    5: 'disgust',
    6: 'angry',
    7: None,
    8: None,
    9: None,
    10: None,
}
```

Ý nghĩa:

- Các nhãn 0-6 được map về 7 class của bài toán.
- Các nhãn 7-10 bị bỏ qua vì không thuộc phạm vi bài toán.

`LABEL_ALIASES` xử lý các tên label khác nhau:

- `anger`, `angriness` -> `angry`
- `happiness` -> `happy`
- `sadness` -> `sad`
- `surprised` -> `surprise`

Các biến tìm cột CSV:

- `AFFECTNET_LABEL_COLUMNS`: các tên cột có thể chứa label.
- `AFFECTNET_PATH_COLUMNS`: các tên cột có thể chứa đường dẫn ảnh.

Cuối cell:

- Set seed cho `random`, `numpy`, `tensorflow`.
- Nếu có GPU và `USE_MIXED_PRECISION=True`, bật mixed precision.
- Nếu không có GPU phù hợp, in `Mixed precision disabled.`

Ý nghĩa trong luồng hoạt động:

- Cell này quyết định toàn bộ cấu hình experiment.
- Các cell xử lý ảnh, split, train model đều dùng biến từ cell này.

## 10. Cell 9 - Tiêu đề tìm hoặc tải dataset

Cell markdown `Locate or Download Dataset`.

Nó báo rằng cell tiếp theo sẽ tìm dataset theo nhiều nguồn khác nhau.

## 11. Cell 10 - Tìm hoặc tải AffectNet

Cell này định nghĩa nhiều hàm để tìm dataset và tạo biến `dataset_roots`.

Các hàm:

### `is_colab_runtime()`

Kiểm tra notebook có chạy trên Google Colab không.

Luồng:

- Thử import `google.colab`.
- Nếu import được thì trả `True`.
- Nếu lỗi thì trả `False`.

### `is_kaggle_runtime()`

Kiểm tra notebook có chạy trên Kaggle không bằng cách xem `/kaggle/input` có tồn tại không.

### `find_kaggle_input_roots()`

Tìm các dataset đã được add vào Kaggle Notebook.

Biến chính:

- `input_dir = Path('/kaggle/input')`
- Trả về danh sách folder con trong `/kaggle/input`.

### `find_local_raw_roots(raw_dir)`

Tìm dataset đã có sẵn trong `data/raw`.

Biến chính:

- `raw_dir`: thường là `RAW_DIR`.
- `roots`: danh sách folder con trong `data/raw`.

### `prepare_kaggle_credentials()`

Chuẩn bị credential Kaggle nếu cần tải dataset.

Luồng:

1. Kiểm tra `~/.kaggle/kaggle.json`.
2. Nếu đã có credential hoặc có env var `KAGGLE_USERNAME`, không làm gì thêm.
3. Nếu đang chạy Kaggle, khuyên dùng Kaggle Input.
4. Nếu đang chạy Colab, yêu cầu upload `kaggle.json`.
5. Nếu không có credential thì báo lỗi.

### `download_kaggle_datasets(dataset_slugs, raw_dir, copy_to_raw=False)`

Đây là hàm chính để quyết định dataset lấy từ đâu.

Thứ tự ưu tiên:

1. Nếu `LOCAL_DATASET_ROOTS` có path, dùng path local.
2. Nếu `data/raw` có folder dataset, dùng lại.
3. Nếu đang chạy Kaggle và có `/kaggle/input`, dùng Kaggle Input.
4. Nếu không có gì, dùng `kagglehub` tải dataset.

Biến quan trọng:

- `dataset_slugs`: danh sách dataset Kaggle.
- `raw_dir`: thư mục `data/raw`.
- `copy_to_raw`: nếu `True`, copy dataset tải về vào `data/raw`.
- `downloaded_paths`: danh sách path dataset tải về.
- `dataset_roots`: kết quả cuối cùng, danh sách root folder dataset.

Ý nghĩa trong luồng hoạt động:

- Sau cell này, notebook biết dataset nằm ở đâu.
- Cell sau sẽ duyệt `dataset_roots` để đọc CSV hoặc ảnh trong folder.

## 12. Cell 11 - Tiêu đề chuẩn hóa label và tạo index ảnh

Cell markdown `Normalize Labels and Build the Index`.

Nó báo rằng cell tiếp theo sẽ:

- Chuẩn hóa label.
- Tìm file ảnh.
- Tạo dataframe `df_images`.

## 13. Cell 12 - Chuẩn hóa label và tạo `df_images`

Đây là một trong những cell quan trọng nhất. Nó biến dataset thô thành dataframe ảnh có path, label và label ID.

Các hàm chính:

### `normalize_label(value, numeric_mapping='affectnet')`

Chuẩn hóa một giá trị label bất kỳ thành một trong 7 class hoặc `None`.

Luồng:

1. Nếu label rỗng hoặc NaN thì trả `None`.
2. Chuyển label thành chữ thường, bỏ khoảng trắng, thay khoảng trắng bằng `_`.
3. Nếu label là các nhãn không dùng như `uncertain`, `non_face`, `contempt`, trả `None`.
4. Nếu label là số, map theo `AFFECTNET_ID_TO_LABEL`.
5. Nếu label là chữ, map theo `LABEL_ALIASES`.

Ví dụ:

- `1` -> `happy`
- `happiness` -> `happy`
- `non_face` -> `None`

### `infer_label_from_path(path)`

Suy ra label từ đường dẫn file hoặc tên folder.

Luồng:

1. Tách path thành từng phần folder/file.
2. Duyệt từ cuối path về đầu.
3. Gọi `normalize_label` trên từng phần.
4. Nếu phần nào map được label thì trả label đó.

Hàm này dùng khi dataset được tổ chức theo folder class.

### `first_matching_column(columns, candidates)`

Tìm cột trong CSV khớp với danh sách tên cột dự kiến.

Ví dụ:

- Tìm cột path trong `AFFECTNET_PATH_COLUMNS`.
- Tìm cột label trong `AFFECTNET_LABEL_COLUMNS`.

### `find_existing_image(root, value)`

Tìm file ảnh thật sự tồn tại dựa trên giá trị path trong CSV.

Luồng:

1. Nếu `value` là path tuyệt đối, thử dùng trực tiếp.
2. Nếu là path tương đối, thử nhiều vị trí:
   - `root / candidate`
   - `root / clean_value`
   - `root / Manually_Annotated / Manually_Annotated_Images / clean_value`
   - `root / Automatically_Annotated / Automatically_Annotated_Images / clean_value`
3. Chỉ nhận file tồn tại và có đuôi nằm trong `IMAGE_EXTS`.

### `collect_affectnet_csv_records(root)`

Đọc các file CSV annotation trong dataset.

Luồng:

1. Duyệt tất cả file `.csv` trong `root`.
2. Đọc thử vài dòng bằng `pd.read_csv(..., nrows=5)`.
3. Tìm cột path và label.
4. Nếu tìm được, đọc CSV theo chunk để tiết kiệm RAM.
5. Với từng dòng:
   - Lấy image path.
   - Chuẩn hóa label.
   - Tìm file ảnh tồn tại.
   - Thêm record vào danh sách.

Mỗi record thường có:

- `path`
- `label`
- `source`
- `annotation_file`

### `collect_folder_label_records(root)`

Thu thập ảnh từ dataset dạng folder class.

Luồng:

1. Duyệt tất cả file ảnh trong `root`.
2. Suy ra label từ path bằng `infer_label_from_path`.
3. Nếu label hợp lệ, thêm record.

### `collect_image_files(dataset_roots)`

Hàm tổng hợp cho nhiều dataset root.

Luồng:

1. Với mỗi root:
   - Ưu tiên đọc CSV annotation bằng `collect_affectnet_csv_records`.
   - Nếu không có record từ CSV, đọc theo folder class bằng `collect_folder_label_records`.
2. Gộp tất cả record.
3. Tạo dataframe `df_images`.
4. Bỏ duplicate path.
5. Bỏ label ngoài `CLASS_NAMES`.
6. Thêm `label_id` theo `CLASS_TO_ID`.

Biến kết quả quan trọng:

- `df_images`: dataframe chính chứa toàn bộ ảnh hợp lệ.

Các cột quan trọng của `df_images`:

- `path`: đường dẫn ảnh.
- `label`: tên class.
- `source`: nguồn dataset.
- `annotation_file`: file CSV annotation nếu có.
- `label_id`: ID số của class.

Kết quả notebook hiện tại:

- Tổng ảnh hợp lệ: `27,755`.
- 7 class được giữ lại.
- Các nhãn không thuộc bài toán bị bỏ.

## 14. Cell 13 - Tiêu đề preview dữ liệu

Cell markdown `Data Preview`.

Nó báo rằng các cell tiếp theo dùng để kiểm tra dữ liệu trước khi train.

## 15. Cell 14 - In thống kê cơ bản của dataset

Cell này in và hiển thị thống kê dataframe `df_images`.

Các lệnh chính:

- In tổng số ảnh:
  ```python
  len(df_images)
  ```
- In số nguồn dataset:
  ```python
  df_images['source'].nunique()
  ```
- In số file ảnh bị thiếu:
  ```python
  (~df_images['path'].map(lambda p: Path(p).exists())).sum()
  ```
- Hiển thị 10 dòng đầu của `df_images`.
- Tạo dataframe `overview` gồm:
  - `count`: số ảnh mỗi class.
  - `percent`: phần trăm mỗi class.

Biến quan trọng:

- `overview`: bảng thống kê class count và percent.

Kết quả hiện tại:

- `Total images`: `27755`
- `Unique sources`: `1`
- `Missing files`: `0`
- Class lớn nhất: `neutral` với `5126` ảnh.
- Class nhỏ nhất: `disgust` với `2477` ảnh.

Ý nghĩa:

- Kiểm tra dataset có đọc đúng không.
- Kiểm tra dữ liệu có bị thiếu file không.
- Nhìn nhanh tình trạng mất cân bằng class.

## 16. Cell 15 - Vẽ biểu đồ phân bố dữ liệu

Cell này visualize dataset bằng 2 biểu đồ.

Nếu `df_images` rỗng:

- In `No images to plot.`

Nếu có dữ liệu:

1. Tạo figure gồm 2 subplot.
2. Vẽ bar chart `Images per Emotion`.
3. Vẽ horizontal bar chart `Top Dataset Sources`.

Biến quan trọng:

- `label_counts`: số ảnh theo từng emotion.
- `source_counts`: số ảnh theo source dataset.

Ý nghĩa:

- Nhìn rõ class imbalance.
- Xem dữ liệu đến từ những source nào.

## 17. Cell 16 - Hiển thị ảnh mẫu theo từng class

Cell này lấy ảnh mẫu từ mỗi class và hiển thị thành grid.

Biến quan trọng:

- `samples_per_class = 4`: lấy tối đa 4 ảnh mỗi class.
- `sample_df`: dataframe ảnh mẫu.
- `labels_present`: danh sách class có ảnh trong sample.
- `rows`: số dòng của grid.
- `cols`: số cột, bằng `samples_per_class`.

Luồng xử lý:

1. Group `df_images` theo `label`.
2. Mỗi class sample tối đa 4 ảnh bằng random seed cố định.
3. Tạo grid với số dòng bằng số class.
4. Mở từng ảnh bằng `Image.open`.
5. Hiển thị ảnh và title label.
6. Nếu ảnh lỗi, hiển thị `Cannot read`.

Ý nghĩa:

- Kiểm tra trực quan ảnh có đúng class không.
- Xem ảnh có đọc được không.
- Nhìn nhanh chất lượng ảnh.

## 18. Cell 17 - Tiêu đề chia train/validation/test

Cell markdown `Train / Validation / Test Split`.

Nó báo rằng cell tiếp theo sẽ chia dataset.

## 19. Cell 18 - Chia dữ liệu train/validation/test

Cell này chia `df_images` thành 3 tập:

- `train_df`
- `valid_df`
- `test_df`

Luồng xử lý:

1. Nếu `df_images` rỗng thì báo lỗi.
2. Kiểm tra mỗi class có ít nhất 2 ảnh để stratified split.
3. Chia lần 1:
   ```python
   train_df, temp_df = train_test_split(...)
   ```
   - Train giữ 70%.
   - `temp_df` giữ 30% còn lại.
   - Dùng `stratify=df_images['label']` để giữ phân bố class.
4. Tính:
   ```python
   relative_test = TEST_SIZE / (VALID_SIZE + TEST_SIZE)
   ```
   Với `VALID_SIZE=0.15`, `TEST_SIZE=0.15`, nên `relative_test=0.5`.
5. Chia `temp_df` thành validation và test.
6. Tạo `split_summary`: số ảnh mỗi class trong train/valid/test.
7. Lưu split CSV:
   - `data/processed/train_split.csv`
   - `data/processed/valid_split.csv`
   - `data/processed/test_split.csv`

Biến quan trọng:

- `min_class_count`: số ảnh của class nhỏ nhất.
- `train_df`: tập train.
- `valid_df`: tập validation.
- `test_df`: tập test.
- `split_summary`: bảng phân bố class sau split.

Kết quả hiện tại:

- Train: `19,428`
- Validation: `4,163`
- Test: `4,164`

Ý nghĩa:

- Tách test set độc lập để đánh giá cuối.
- Validation dùng trong training để chọn checkpoint tốt nhất.
- Stratified split giúp mỗi tập giữ phân bố class gần giống nhau.

## 20. Cell 19 - Tiêu đề TensorFlow Dataset

Cell markdown `TensorFlow Datasets`.

Nó báo rằng cell tiếp theo sẽ biến dataframe thành `tf.data.Dataset`.

## 21. Cell 20 - Tạo TensorFlow Dataset và class weights

Cell này chuyển `train_df`, `valid_df`, `test_df` thành dataset dùng cho TensorFlow.

Biến:

- `AUTOTUNE = tf.data.AUTOTUNE`: để TensorFlow tự tối ưu parallel/prefetch.

### Hàm `load_image(path, label)`

Đọc và xử lý một ảnh.

Luồng:

1. Đọc file bằng `tf.io.read_file`.
2. Decode ảnh bằng `tf.io.decode_image`, ép 3 channel RGB.
3. Set shape `[None, None, 3]`.
4. Resize về `IMG_SIZE`, tức `224x224`.
5. Cast sang `tf.float32`.
6. Trả về `(image, label)`.

Lưu ý:

- Cell này chỉ resize và cast float.
- MobileNetV2 preprocessing được áp dụng trong model ở cell 22.

### Hàm `make_dataset(dataframe, training=False)`

Tạo `tf.data.Dataset` từ dataframe.

Luồng:

1. Lấy `paths` từ cột `path`.
2. Lấy `labels` từ cột `label_id`.
3. Tạo dataset bằng `tf.data.Dataset.from_tensor_slices`.
4. Nếu là training set:
   - Shuffle dữ liệu.
5. Map qua `load_image`.
6. Batch theo `BATCH_SIZE`.
7. Prefetch để tăng tốc training.

Dataset được tạo:

- `train_ds = make_dataset(train_df, training=True)`
- `valid_ds = make_dataset(valid_df)`
- `test_ds = make_dataset(test_df)`

Class weight:

```python
class_weights_arr = compute_class_weight(...)
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
```

Ý nghĩa:

- Class ít ảnh hơn sẽ có trọng số cao hơn trong loss.
- Giúp giảm bias của model về class nhiều ảnh.

Kết quả class weight hiện tại:

- `angry`: khoảng `1.2319`
- `disgust`: khoảng `1.6006`
- `fear`: khoảng `1.2485`
- `happy`: khoảng `0.7860`
- `neutral`: khoảng `0.7735`
- `sad`: khoảng `0.8482`
- `surprise`: khoảng `0.9818`

## 22. Cell 21 - Tiêu đề build và train model

Cell markdown `Build and Train the Model`.

Nó báo rằng cell tiếp theo sẽ:

- Tạo model MobileNetV2.
- Train classification head.
- Fine-tune backbone.

## 23. Cell 22 - Build MobileNetV2, train head và fine-tune

Đây là cell training chính.

### Data augmentation

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name='augmentation')
```

Ý nghĩa:

- Tăng biến thể dữ liệu train.
- Giúp model generalize tốt hơn.
- Các phép biến đổi gồm flip ngang, xoay nhẹ, zoom nhẹ, đổi contrast nhẹ.

### Backbone MobileNetV2

```python
base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=IMG_SIZE + (3,),
)
```

Biến:

- `include_top=False`: bỏ classifier gốc của ImageNet.
- `weights='imagenet'`: dùng pretrained weights.
- `input_shape=(224, 224, 3)`: ảnh RGB 224x224.

Ban đầu:

```python
base_model.trainable = False
```

Tức là freeze MobileNetV2, chỉ train phần head.

### Classification head

Luồng model:

```text
Input 224x224x3
-> data_augmentation
-> MobileNetV2 preprocess_input
-> MobileNetV2 backbone
-> GlobalAveragePooling2D
-> Dropout(0.35)
-> Dense(7, softmax)
```

Các biến:

- `inputs`: input tensor.
- `x`: tensor trung gian.
- `outputs`: output xác suất 7 class.
- `model`: Keras model hoàn chỉnh.

Layer cuối:

```python
layers.Dense(len(CLASS_NAMES), activation='softmax', dtype='float32')
```

Ý nghĩa:

- Output có 7 giá trị xác suất.
- `dtype='float32'` giúp ổn định nếu dùng mixed precision.

### Compile lần 1 - train head

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

Thông số:

- Optimizer: Adam.
- Learning rate: `1e-3`.
- Loss: `sparse_categorical_crossentropy`.
- Metric: accuracy.

### Callbacks

```python
checkpoint_path = MODEL_DIR / 'best_emotion_model.keras'
```

Callbacks:

- `ModelCheckpoint`: lưu model tốt nhất theo `val_accuracy`.
- `EarlyStopping`: dừng sớm nếu `val_accuracy` không cải thiện 5 epoch, restore best weights.
- `ReduceLROnPlateau`: giảm LR nếu `val_loss` không cải thiện.

### Train head

```python
history_head = model.fit(...)
```

Dữ liệu:

- Train: `train_ds`.
- Validation: `valid_ds`.
- Epoch: `EPOCHS_HEAD = 8`.
- Dùng `class_weight=class_weights`.

Ý nghĩa:

- Vì backbone đang freeze, giai đoạn này chỉ học classification head.

### Fine-tune

Sau train head:

```python
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.70)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

Ý nghĩa:

- Unfreeze MobileNetV2.
- Nhưng giữ frozen 70% layer đầu.
- Chỉ fine-tune khoảng 30% layer cuối.

Compile lần 2:

- Learning rate giảm xuống `1e-5`.
- Loss và metric giữ nguyên.

Train fine-tune:

```python
history_fine = model.fit(...)
```

Biến quan trọng:

- `data_augmentation`
- `base_model`
- `model`
- `checkpoint_path`
- `callbacks`
- `history_head`
- `fine_tune_at`
- `history_fine`

Ý nghĩa trong toàn pipeline:

- Đây là phần tạo model và học từ dữ liệu.
- Checkpoint tốt nhất được lưu tại `artifacts/models/best_emotion_model.keras`.

## 24. Cell 23 - Tiêu đề evaluate và lưu artifact

Cell markdown `Evaluate and Save Artifacts`.

Nó báo rằng cell tiếp theo sẽ:

- Load checkpoint tốt nhất.
- Evaluate trên test set.
- Vẽ report, confusion matrix, learning curves.
- Lưu model cuối và metadata.

## 25. Cell 24 - Evaluate model và lưu artifact

Cell này đánh giá model và lưu kết quả quan trọng.

### Hàm `merge_history(*histories)`

Gộp history của 2 giai đoạn:

- `history_head`
- `history_fine`

Luồng:

1. Tạo dict `merged`.
2. Với từng history, duyệt các key như `accuracy`, `val_accuracy`, `loss`, `val_loss`.
3. Nối các list metric lại với nhau.

Mục đích:

- Vẽ learning curve liên tục cho cả 2 giai đoạn train.

### Load best model

```python
best_model = keras.models.load_model(checkpoint_path)
```

Notebook dùng checkpoint tốt nhất theo validation accuracy, không nhất thiết dùng weights cuối cùng.

### Evaluate test set

```python
test_loss, test_acc = best_model.evaluate(test_ds)
```

In:

- `Test loss`
- `Test accuracy`

Theo metadata hiện tại:

- `test_accuracy = 0.6174351572990417`
- Tương đương khoảng `61.74%`.

### Tạo prediction cho test set

Biến:

- `y_true`: label thật.
- `y_pred`: label model dự đoán.
- `probs`: xác suất dự đoán cho mỗi batch.

Luồng:

1. Duyệt từng batch trong `test_ds`.
2. Dùng `best_model.predict`.
3. Lấy class dự đoán bằng `np.argmax`.
4. Lưu label thật và label dự đoán.

### Classification report

```python
classification_report(y_true, y_pred, target_names=CLASS_NAMES)
```

Report gồm:

- precision
- recall
- f1-score
- support

Ý nghĩa:

- Cho biết từng class hoạt động tốt/yếu thế nào.
- Quan trọng hơn accuracy khi dataset mất cân bằng.

### Confusion matrix

```python
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
```

Ý nghĩa:

- Cho biết class nào hay bị nhầm sang class nào.

### Learning curves

Dùng `hist = merge_history(history_head, history_fine)` để vẽ:

- Train accuracy.
- Validation accuracy.
- Train loss.
- Validation loss.

Ý nghĩa:

- Kiểm tra overfitting/underfitting.
- Xem quá trình train có cải thiện không.

### Lưu model cuối

```python
final_model_path = MODEL_DIR / 'face_emotion_mobilenetv2_v2.keras'
best_model.save(final_model_path)
```

Model cuối được lưu tại:

```text
artifacts/models/face_emotion_mobilenetv2_v2.keras
```

### Lưu metadata

Metadata gồm:

- `class_names`
- `img_size`
- `kaggle_datasets`
- `dataset_roots`
- `train_samples`
- `valid_samples`
- `test_samples`
- `test_accuracy`
- `class_weights`

File metadata:

```text
artifacts/models/metadata_v2.json
```

Ý nghĩa:

- Dùng lại khi inference.
- Giữ thông tin label order, input size và accuracy.

## 26. Cell 25 - Tiêu đề predict ảnh đơn và face detection

Cell markdown `Single-Image Prediction and Face Detection`.

Nó báo rằng các cell cuối dùng để demo inference.

## 27. Cell 26 - Load model và metadata trong Colab

Cell này dành cho trường hợp chạy inference/demo trong Google Colab.

Các import:

- `files` từ `google.colab`
- `keras`
- `json`

Biến:

- `model_path = "/face_emotion_mobilenetv2_v2.keras"`
- `best_model`: model load lại từ file `.keras`.
- `metadata_file`: file metadata upload từ máy.
- `metadata`: nội dung JSON metadata.
- `CLASS_NAMES`: lấy lại từ metadata.
- `IMG_SIZE`: lấy lại từ metadata.

Luồng:

1. In `keras`.
2. Load model từ `model_path`.
3. Upload metadata bằng `files.upload()`.
4. Đọc metadata JSON.
5. Gán lại `CLASS_NAMES` và `IMG_SIZE`.
6. In thông tin model/metadata đã load.

Lưu ý:

- Cell này phụ thuộc Colab.
- Đường dẫn `"/face_emotion_mobilenetv2_v2.keras"` là đường dẫn kiểu Colab/root, không phải đường dẫn project local.
- Nếu chạy local, nên chỉnh path về `artifacts/models/face_emotion_mobilenetv2_v2.keras`.

## 28. Cell 27 - Upload ảnh, predict cảm xúc và detect face

Cell này dùng để demo inference trên ảnh upload.

Các import:

- `files` từ `google.colab`
- `Image` từ PIL
- `numpy`
- `pandas`
- `matplotlib.pyplot`
- `cv2`

Đầu cell có một block code cũ bị comment. Phần code chạy thật nằm phía dưới.

### Upload ảnh

```python
uploaded_image = files.upload()
image_path = list(uploaded_image.keys())[0]
```

Biến:

- `uploaded_image`: dict chứa file upload.
- `image_path`: tên file ảnh đầu tiên upload.

### Hàm `predict_emotion(image_path, model=best_model)`

Predict cảm xúc cho một ảnh.

Luồng xử lý:

1. Mở ảnh bằng PIL.
2. Convert ảnh sang RGB.
3. Resize về `IMG_SIZE`.
4. Chuyển ảnh thành NumPy array `float32`.
5. Thêm batch dimension bằng `[None, ...]`.
6. Gọi `model.predict`.
7. Lấy class có xác suất cao nhất bằng `np.argmax`.
8. Hiển thị ảnh với title gồm label và confidence.
9. Trả về dataframe gồm:
   - `emotion`
   - `probability`
   được sort giảm dần theo probability.

Biến quan trọng:

- `image`
- `array`
- `probs`
- `pred_idx`
- `pred_label`
- `confidence`
- `result_df`

### Haar Cascade face detector

Cell dùng:

```python
face_cascade = cv2.CascadeClassifier(...)
```

Mục đích:

- Detect khuôn mặt trong ảnh bằng Haar Cascade.

### Hàm `detect_faces_and_predict(image_path, model=best_model)`

Predict cảm xúc cho từng khuôn mặt trong ảnh.

Luồng xử lý:

1. Đọc ảnh bằng `cv2.imread`.
2. Nếu không đọc được ảnh thì báo lỗi.
3. Convert ảnh sang grayscale.
4. Detect faces bằng:
   ```python
   face_cascade.detectMultiScale(...)
   ```
5. Với mỗi face bounding box `(x, y, w, h)`:
   - Crop vùng mặt.
   - Convert BGR sang RGB.
   - Resize về `IMG_SIZE`.
   - Predict bằng model.
   - Lấy label và confidence.
   - Vẽ rectangle quanh mặt.
   - Ghi text label lên ảnh.
   - Lưu kết quả vào list.
6. Convert ảnh kết quả sang RGB.
7. Hiển thị ảnh bằng matplotlib.
8. Trả về dataframe kết quả từng face.

Biến quan trọng:

- `bgr`: ảnh đọc bằng OpenCV.
- `gray`: ảnh grayscale để detect face.
- `faces`: danh sách bounding box.
- `results`: danh sách kết quả từng khuôn mặt.
- `face`: ảnh crop khuôn mặt.
- `face_rgb`: crop đã convert RGB và resize.
- `results_df`: dataframe kết quả.

Cuối cell thường chạy:

- `single_result = predict_emotion(image_path)`
- `face_result = detect_faces_and_predict(image_path)`

Ý nghĩa:

- `predict_emotion` dùng toàn bộ ảnh.
- `detect_faces_and_predict` detect mặt rồi predict từng mặt.

Lưu ý kỹ thuật:

- Training resize ảnh dataset trực tiếp.
- Demo multi-face lại crop face trước khi predict.
- Nếu dữ liệu train và ảnh inference có cách crop khác nhau, phân phối input có thể lệch.
- Haar Cascade đơn giản, có thể yếu với ảnh tối, mặt nghiêng, che mặt hoặc mặt nhỏ.

## 29. Các artifact được tạo ra

Sau khi chạy notebook đầy đủ, các file quan trọng gồm:

```text
data/processed/train_split.csv
data/processed/valid_split.csv
data/processed/test_split.csv
artifacts/models/best_emotion_model.keras
artifacts/models/face_emotion_mobilenetv2_v2.keras
artifacts/models/metadata_v2.json
```

Ý nghĩa:

- `train_split.csv`, `valid_split.csv`, `test_split.csv`: lưu lại dữ liệu đã chia.
- `best_emotion_model.keras`: checkpoint tốt nhất theo validation accuracy.
- `face_emotion_mobilenetv2_v2.keras`: model cuối để dùng inference.
- `metadata_v2.json`: thông tin class, input size, split size, test accuracy và class weight.

## 30. Luồng dữ liệu chi tiết

```text
Kaggle/local dataset
-> dataset_roots
-> collect_affectnet_csv_records hoặc collect_folder_label_records
-> df_images
-> train_df, valid_df, test_df
-> train_ds, valid_ds, test_ds
-> MobileNetV2 model
-> history_head, history_fine
-> best_model
-> test metrics
-> saved model + metadata
-> inference functions
```

Trong đó:

- `dataset_roots` là danh sách thư mục nguồn.
- `df_images` là bảng ảnh hợp lệ.
- `train_df`, `valid_df`, `test_df` là dữ liệu đã split.
- `train_ds`, `valid_ds`, `test_ds` là TensorFlow Dataset.
- `model` là model đang train.
- `best_model` là model load từ checkpoint tốt nhất.

## 31. Những điểm notebook đã làm tốt

- Có pipeline hoàn chỉnh từ dataset đến model artifact.
- Có xử lý nhiều kiểu dataset: CSV annotation hoặc folder class.
- Có chuẩn hóa label AffectNet về 7 class.
- Có preview dữ liệu và visualize class distribution.
- Có stratified split.
- Có class weight cho class imbalance.
- Có transfer learning bằng MobileNetV2.
- Có 2 giai đoạn train: train head và fine-tune backbone.
- Có callbacks để lưu checkpoint và giảm overfitting.
- Có evaluate bằng accuracy, classification report, confusion matrix.
- Có lưu model và metadata.
- Có demo inference cho ảnh đơn và multi-face.

## 32. Những điểm còn hạn chế

- Notebook chưa có phần Related Work. Phần này đang nằm trong report.
- Chưa lưu `classification_report`, confusion matrix và learning curves thành file riêng.
- Chưa kiểm tra ảnh hỏng bằng một bước scan toàn dataset trước train.
- Chưa kiểm tra duplicate ảnh bằng hash.
- Chưa kiểm tra ảnh không có khuôn mặt.
- Chưa có so sánh với custom CNN, EfficientNet, ResNet hoặc ConvNeXt.
- Cell inference cuối phụ thuộc Google Colab và cần chỉnh path nếu chạy local.

## 33. Tóm tắt ngắn gọn

Notebook này là một baseline đầy đủ cho bài toán facial emotion recognition:

- Dataset: AffectNet.
- Task: classification 7 emotion classes.
- Model: MobileNetV2 pretrained ImageNet + custom softmax head.
- Training: train head rồi fine-tune backbone.
- Evaluation: test accuracy, classification report, confusion matrix.
- Kết quả chính: khoảng `61.74%` test accuracy.
- Artifact: model `.keras` và `metadata_v2.json`.

Về mặt yêu cầu implementation, notebook đã đủ tốt cho một baseline. Phần cần bổ sung nếu muốn hoàn chỉnh hơn là data quality sâu hơn, lưu metric ra file và thêm so sánh model.
