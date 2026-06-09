# 🧬 VaccineNLP: Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam
*(Applying Natural Language Processing for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)*

---

## 🏛️ ĐỒ ÁN TỐT NGHIỆP CỬ NHÂN KHOA HỌC DỮ LIỆU (HUPH 2026)
* **Sinh viên thực hiện:** 
  * Kim Mạnh Hưng (MSSV: 2211090016)
  * Đinh Lê Quỳnh Phương (MSSV: 2211090031)
* **Giảng viên hướng dẫn:** TS. Trần Lâm Quân
* **Cơ sở đào tạo:** Trường Đại học Y tế Công cộng (HUPH)

---

## 🔗 Liên kết tài nguyên dự án (Resource Registry)

### 🤖 Models trên Hugging Face
* [![Model: PhoBERT-Multitask](https://img.shields.io/badge/Small_Model-PhoBERT--Multitask-orange?style=flat-square)](https://huggingface.co/hung2903/phobert-vaccine-multitask) — *Phân loại đa nhiệm tốc độ cao.*
* [![Model: Gemma-4-E4B-it](https://img.shields.io/badge/Reasoning_Engine-Gemma--4--E4B--it-blue?style=flat-square)](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai) — *Động cơ lý luận Chain-of-Thought giải thích y khoa.*
* [![Model: XLMR-Multitask](https://img.shields.io/badge/Small_Model-XLMR--Multitask-red?style=flat-square)](https://huggingface.co/hung2903/xlmr-vaccine-multitask) — *Mô hình phân loại Baseline đa ngôn ngữ.*

### 🌐 Ứng dụng Demo trực tuyến (Spaces Demo)
* **Hưng Space (Official Demo):** [hung2903/vaccinenlp-demo](https://huggingface.co/spaces/hung2903/vaccinenlp-demo)
* **Phương Space (Redesigned Interface):** [quynhphuong1209/VaccineNLP_demo](https://huggingface.co/spaces/quynhphuong1209/VaccineNLP_demo), (https://huggingface.co/spaces/quynhphuong1209/VaccineNLP_Ban_chinh_thuc)

---

## ️ Cấu Trúc & Tổ Chức Workspace (Workspace Organization)

### 📁 Cấu Trúc Thư Mục Chính

```
VaccineNLP_Clean_V1/
├── 📁 app/                      # Streamlit demo application
│   ├── streamlit_demo.py        # Main UI (3600 LOC)
│   ├── xai_cache.json           # Shared XAI reasoning cache
│   └── requirements_demo.txt
│
├── 📁 app_gradio/               # Gradio demo application  
│   ├── app.py                   # Main UI (2800 LOC)
│   ├── gradio_app.py
│   └── requirements.txt
│
├── 📁 src/                      # Core modules
│   ├── 📁 app_core/             # ⭐ NEW: Shared logic module (1200+ LOC)
│   │   ├── __init__.py          # Module exports
│   │   ├── predictor.py         # Model loading & inference (450 LOC)
│   │   ├── xai_engine.py        # XAI reasoning & generation (400 LOC)
│   │   ├── fetchers.py          # URL data fetching (350 LOC)
│   │   └── README.md
│   │
│   ├── 📁 preprocessing/        # Text cleaning & tokenization
│   ├── 📁 modeling/             # Model training & inference
│   ├── 📁 common/               # Utilities & paths
│   └── README.md
│
├── 📁 datasets/                 # Data with Medallion architecture
│   ├── 01_raw/                  # Bronze: Raw scraped data
│   ├── 02_interim/              # Silver: Work-in-progress
│   ├── 02_processed/            # Silver: Cleaned, unlabeled
│   ├── 04_silver_labels/        # Silver: Auto-labeled
│   ├── 03_processed/            # Gold: Expert-reviewed
│   ├── 05_model_ready/          # Execution: Train/Val/Test splits
│   ├── temp/                    # Temporary working files
│   └── README.md                # Medallion architecture guide
│
├── 📁 notebooks/                # Jupyter notebooks
│   ├── 01_vaccinenlp-phobert-v2-multitask.ipynb
│   ├── 02_vaccinenlp-xlm-r-v1-multitask-classifier.ipynb
│   ├── 03A_vaccinenlp-gemma-4-training.ipynb
│   ├── 03B_vaccinenlp-gemma-4-inference.ipynb
│   ├── 04_vaccinenlp-model-benchmark-report.ipynb
│   └── 05_vaccinenlp-hypothesis-testing.ipynb
│
├── 📁 experiments/              # Model checkpoints & results
│   ├── models/                  # Trained model weights
│   └── results/                 # Benchmark results & metrics
│
├── 📁 docs/                     # Comprehensive documentation
│   ├── 01_PIPELINE_ARCHITECTURE.md
│   ├── 02_DATASET_CARD.md
│   ├── 03_METHODOLOGY.md
│   ├── MODULES_DOCUMENTATION.md  # ⭐ NEW API documentation
│   ├── DEPRECATIONS.md           # ⭐ NEW cleanup record
│   └── [more guides]
│
├── 📁 configs/                  # Configuration files
│   ├── taxonomy.json            # Label definitions
│   ├── class_weights_v2.json    # Class imbalance weights
│   └── seeds.json
├── README.md                    # This file
└── requirements.txt             # Global dependencies
```

## 📖 TÓM TẮT NỘI DUNG NGHIÊN CỨU KHOA HỌC (THESIS BRIEFING)

```text
📥 Dữ liệu Mạng Xã Hội Thô (Bronze)
 └── 🧹 Làm sạch & Chuẩn hóa từ lóng ──> Silver Raw
       └── 🤖 Gán nhãn yếu bằng Gemma 31B Oracle
             └── ✍️ Kiểm duyệt bởi Tác giả (HITL)
                   ├── 🎯 Tách tập Gold Test Set độc lập (n=186)
                   └── 🧠 Huấn luyện Mô hình Đa nhiệm & QLoRA
                         ├── 🔸 PhoBERT-v2 Classifier
                         │     └── 🌡️ Hiệu chuẩn Temperature Scaling ──> Hệ thống Hybrid
                         └── 🔹 Gemma-4-E4B-it XAI Reasoner ───────────/   (Dual-Student)
                                                                            │
                                                                            └──> 📊 Trực quan hóa
                                                                                 (Integrated Gradients & CoT)
```

### 1. 🚨 ĐẶT VẤN ĐỀ (INTRODUCTION)
Trong kỷ nguyên số, mạng xã hội đã trở thành một kênh truyền thông y tế có sức lan tỏa mạnh mẽ nhưng cũng là nơi phát tán hàng loạt **thông tin sai lệch về vắc-xin (vaccine misinformation)**. Việc này thúc đẩy tâm lý **tiêm chủng do dự (vaccine hesitancy)** hoặc lập trường phản đối tiêm chủng trong cộng đồng, ảnh hưởng tiêu cực đến mục tiêu đạt miễn dịch cộng đồng tại Việt Nam.

**Những thách thức chính hiện nay:**
* **Ngụy trang ngôn ngữ (Linguistic Camouflage):** Các nhóm lập trường phản đối tiêm chủng thường xuyên sử dụng các thuật từ lóng, mật ngữ hoặc từ nói giảm nói tránh (vd: *"nước cất", "sinh tố", "chốt", "rụng lá"*) nhằm tránh các bộ lọc tự động của mạng xã hội.
* **Hộp đen AI (Black-Box Problem):** Các mô hình Deep Learning truyền thống chỉ đưa ra nhãn phân loại (vd: *Tin giả, Tiêu cực*) mà không thể giải thích **Tại sao**, khiến các chuyên gia dịch tễ học và bác sĩ chưa có đủ cơ sở tin tưởng vào kết quả phân tích.
* **Mất hiệu chuẩn độ tin cậy (Miscalibration):** Các mạng neural sâu hiện đại thường bị hiện tượng "quá tự tin" (xác suất dự báo cao nhưng thực tế độ chính xác thấp), gây rủi ro khi ứng dụng vào y tế công cộng.

**Mục tiêu đề tài:** Xây dựng hệ thống Hybrid tiên phong phối hợp giữa mô hình phân loại đa nhiệm tiếng Việt chuyên sâu và mô hình ngôn ngữ lớn (LLM) để vừa phát hiện tin sai lệch, phân tích thái độ đa chiều, vừa cung cấp khả năng giải thích khoa học dựa trên bằng chứng (XAI).

---

### 2. 📚 TỔNG QUAN TÀI LIỆU (LITERATURE REVIEW)
Nghiên cứu được thiết lập dựa trên việc kế thừa và phát triển từ các công trình khoa học tiền đề:
* **MiSoVac:** Bộ dữ liệu đa ngôn ngữ về tin sai lệch vắc-xin, làm cơ sở so sánh tính đặc thù ngữ cảnh của tiếng Việt đối với các mật ngữ bản địa hóa.
* **Lý thuyết miscalibration (Guo et al., ICML 2017):** Khẳng định các mô hình Transformer hiện đại có xu hướng phân phối xác suất cực đoan (softmax confidence tiệm cận 1.0). Đề tài kế thừa giải pháp **Temperature Scaling** để đưa xác suất dự đoán về đúng với tần suất chính xác thực tế.
* **Integrated Gradients (Sundararajan et al.):** Cơ sở lý thuyết của Explainable AI (XAI) nhằm tính toán đóng góp thuộc tính của từng từ (token attribution) đối với đầu ra dự đoán.

---

### 3. 🧪 PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)

Dự án được triển khai qua **kiến trúc Medallion 5 giai đoạn chặt chẽ**:

#### A. Quy trình Xử lý Dữ liệu & Quy mô Dữ liệu
* **Tập Bronze (Thô):** Dữ liệu thu thập đa nguồn từ Facebook, YouTube, TikTok bằng Apify Client & yt-dlp.
* **Tập Silver Raw (1.856 mẫu):** Đã đi qua **Triple Filter Pipeline** (Chuẩn hóa Unicode, Dịch Teen-code, Lọc domain y tế loại bỏ vắc-xin thú y).
* **Tập Silver Train/Val (1.663 mẫu):** Dữ liệu gán nhãn yếu bởi mô hình Oracle Gemma 31B. Trong đó:
  * **90% Train (1.496 mẫu):** Dùng để huấn luyện trọng số cho mô hình.
  * **10% Val (167 mẫu):** Dùng để theo dõi Overfitting (Early stopping) và tối ưu hóa tham số hiệu chuẩn nhiệt độ.
* **Tập Gold Test Set (186 mẫu độc lập):** Áp dụng quy trình kiểm duyệt mù **Human-in-the-Loop (HITL)** bởi tác giả đề tài (con người kiểm duyệt) để tạo ra tập dữ liệu kiểm thử chuẩn phục vụ đánh giá độc lập (Ground Truth). 
  * Hệ số đồng thuận **Cohen's Kappa** trung bình đạt **0.2711** (Kappa của trục Misinfo đạt giá trị nhỏ nhất là 0.0525), chứng minh thực nghiệm luận điểm: *AI chưa thể thay thế hoàn toàn con người trong việc xác định tính chính xác của tri thức y khoa phức tạp.*

#### B. Động cơ Phân loại Đa nhiệm (Classification Engine)
* Tinh chỉnh kiến trúc Multi-task Learning trên nền pre-trained **PhoBERT-base-v2**.
* Thiết kế 1 Encoder chung kết hợp **3 đầu ra song song (Heads)**:
  1. Phát hiện Tin sai lệch (Misinfo) - 2 nhãn.
  2. Phân tích Lập trường (Stance) - 3 nhãn model-ready (Ủng hộ, Phản đối, Trung lập).
  3. Phân tích Cảm xúc (Sentiment) - 3 nhãn (Tích cực, Tiêu cực, Trung tính).
* Áp dụng **Weighted CrossEntropy Loss** có tính đến trọng số nghịch đảo tần suất lớp (`class_weight='balanced'`) để triệt tiêu ảnh hưởng của hiện tượng mất cân bằng dữ liệu nghiêm trọng.

#### C. Động cơ Lý luận & Giải thích (XAI Reasoning Engine)
* Huấn luyện mô hình **Gemma-4-E4B-it QLoRA (Nf4 double quantization)** thông qua framework **Unsloth**.
* Mô hình được học cách sinh ra chuỗi lập luận tự nhiên (**Chain-of-Thought**) dựa trên các Rationale được sinh từ mô hình Oracle và hiệu chỉnh bởi con người kiểm duyệt.
* Áp dụng cơ chế **Robust Parsing** kết hợp **Raw Response Logging** lưu trữ phản hồi thô để theo dõi và khắc phục tỷ lệ lỗi sinh định dạng.

#### D. Hiệu chuẩn Độ tin cậy (Confidence Calibration)
Áp dụng công thức post-hoc chuẩn y tế:
```
f_i(x) = softmax(z_i / T)
```
Với các tham số nhiệt độ T được cực tiểu hóa hàm lỗi Negative Log-Likelihood (NLL) bằng thuật toán L-BFGS trên Validation set:
* $T_{\text{misinfo}} = 1.8197$
* $T_{\text{stance}} = 1.6666$
* $T_{\text{sentiment}} = 1.3474$

---

### 4. 📊 KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS)

#### 📊 A. Hiệu Năng Macro F1 Tổng Thể (Macro F1 Summary)
Dưới đây là kết quả đánh giá cuối cùng của các mô hình trên tập **Gold Test (186 mẫu)**:

| Mô hình | Loại kiến trúc | Misinfo (F1) | Stance (F1) | Sentiment (F1) | Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PhoBERT-v2** | Classification Engine (Encoder) | 0.6996 | **0.6640** | 0.7266 | **SOTA Classification Engine** |
| **Gemma-4-E4B-it** | XAI Reasoning Engine (Decoder) | 0.6377 | 0.6264 | **0.7700** | **SOTA Sentiment / XAI Engine** |
| **XLM-R-v1** | Baseline (Encoder) | **0.7038** | 0.6224 | 0.6866 | Baseline phân loại |

* **Đánh giá ECE (Expected Calibration Error):** 
  * Trục Misinfo (T=1.82): ECE giảm từ **12.3%** xuống còn **5.4%** (cải thiện **56%** độ tin cậy).
  * Trục Stance (T=1.67): ECE giảm từ **19.8%** xuống còn **9.3%** (cải thiện **53%**).
  * Trục Sentiment (T=1.35): ECE giảm từ **14.4%** xuống còn **8.1%** (cải thiện **44%**).
* **Đánh giá rủi ro LLM:** Mô hình giải thích Gemma-4-E4B-it đạt tỷ lệ **Parse Failure Rate là 28.0%** trên tập Gold Test (do sinh câu mào đầu ngoài lề hoặc hỏng cấu trúc phân tách), minh chứng cho độ bất ổn định về định dạng đầu ra của mô hình Decoder kích thước nhỏ trong môi trường sản xuất thực tế nếu không được thiết lập cơ chế kiểm soát cú pháp chặt chẽ.

#### 🔍 B. Đánh Giá Chi Tiết Theo Từng Nhãn Lớp (Per-Class F1 breakdown)

##### 1. Trục Phát hiện Tin sai lệch (Misinformation)
* **Tin giả (Misinfo):** XLM-R: **0.5079** | PhoBERT: **0.5075** | Gemma-4-E4B-it: **0.4444** *(Support Gold: 28)*
* **Chính xác (Correct):** XLM-R: **0.8997** | PhoBERT: **0.8918** | Gemma-4-E4B-it: **0.8309** *(Support Gold: 158)*

##### 2. Trục Phân tích Lập trường (Stance)
* **Ủng hộ (Support):** XLM-R: 0.5495 | PhoBERT: **0.5934** | Gemma-4-E4B-it: 0.4528 *(Support Gold: 54)*
* **Phản đối (Against):** XLM-R: 0.6387 | PhoBERT: 0.6612 | Gemma-4-E4B-it: **0.6905** *(Support Gold: 48)*
* **Trung lập (Neutral):** XLM-R: 0.6790 | PhoBERT: **0.7375** | Gemma-4-E4B-it: 0.7360 *(Support Gold: 84)*

##### 3. Trục Phân tích Cảm xúc (Sentiment)
* **Tiêu cực (Negative):** XLM-R: 0.7682 | PhoBERT: 0.8000 | Gemma-4-E4B-it: **0.8039** *(Support Gold: 71)*
* **Trung tính (Neutral):** XLM-R: 0.7162 | PhoBERT: 0.7917 | Gemma-4-E4B-it: **0.8034** *(Support Gold: 75)*
* **Tích cực (Positive):** XLM-R: 0.5753 | PhoBERT: 0.5882 | Gemma-4-E4B-it: **0.7027** *(Support Gold: 40)*

---

### 5. 💬 BÀN LUẬN (DISCUSSION)

#### A. Kiến trúc Hybrid Dual-Student mang lại giá trị thực tiễn vượt trội
Nghiên cứu chỉ ra sự phân hóa ưu thế kiến trúc rõ rệt:
* **Mô hình Encoder (PhoBERT-v2)** cho độ ổn định cao hơn, tốc độ xử lý nhanh, F1 phân loại Stance tốt nhất (0.6640).
* **Mô hình Decoder (Gemma-4-E4B-it)** mạnh hơn về phân tích cảm xúc (Sentiment F1 = 0.7700) nhờ khả năng nắm bắt ngữ cảnh sinh văn bản tinh tế, đồng thời cung cấp chuỗi lý giải y học minh bạch.
* Phối hợp hai mô hình tạo ra hệ thống **Dual-Student Hybrid** lý tưởng: PhoBERT đưa ra dự đoán nhãn nhanh chóng và chính xác vượt trội, trong khi Gemma cung cấp lý luận y khoa dễ hiểu cho người dùng.

#### B. Phân tích rủi ro truyền thông dịch tễ học qua kiểm định giả thuyết thống kê (H1-H4)
Các kiểm định thống kê trên tập Gold Test (186 mẫu) mang lại những phát hiện định lượng có ý nghĩa thực tiễn cho y tế công cộng:
* **Cảm xúc gói (Emotion Packaging - H1):** Lập trường phản đối vắc-xin có mối liên hệ thống kê chặt chẽ với cảm xúc tiêu cực (p = 6.85 x 10^-40). **93.8%** các bài viết phản đối vắc-xin mang sắc thái cảm xúc tiêu cực (lo lắng, sợ hãi, giận dữ).
* **Platform-specific Risk (H3):** Tin sai lệch phân bổ không đồng đều giữa các nền tảng (p = 0.0021). **Facebook (24.7%)** và **YouTube (12.2%)** là các nền tảng ghi nhận tỷ lệ phân bổ tin giả lớn, trong khi báo chí chính thống và diễn đàn học thuật đạt tỷ lệ 0.0%.
* **Sự lồng ghép lập trường phản đối và tin giả (H4):** **50.0%** nội dung có lập trường phản đối vắc-xin chứa đựng tin sai lệch về mặt y khoa (24/48 mẫu), trong khi con số này ở nhóm ủng hộ chỉ là **1.9%** (1/54 mẫu). Điều này khẳng định tin giả là phương thức chủ yếu được sử dụng để xây dựng lập luận chống tiêm chủng.

#### C. Độ phức tạp của bài toán phát hiện Tin giả (F1 Misinfo ~ 0.50)
F1-score cho lớp thiểu số "Tin giả" của mọi mô hình chỉ đạt quanh mức **0.50** (dù F1 của nhãn "Chính xác" là ~0.89). Điều này phản ánh đặc trưng ngụy trang ngôn ngữ của các nhóm lập trường phản đối tiêm chủng tại Việt Nam. Họ sử dụng ẩn dụ hóa y học để tránh các bộ lọc từ khóa, đặt ra yêu cầu AI phải học được các chuỗi lý luận phi tuyến tính.

---

### 6. 🏁 KẾT LUẬN (CONCLUSION)

#### A. Đóng góp khoa học chính của đề tài
1. **Kiến trúc Hybrid Tiên phong:** Kết hợp thành công thế mạnh của SLM Encoder chuyên biệt (PhoBERT-v2) và Generator LLM QLoRA (Gemma-4-E4B-it) để giải quyết song song bài toán phân loại và giải thích y khoa.
2. **Quy trình HITL & Bộ kiểm thử Gold Test cực chuẩn:** Thiết lập hệ thống dữ liệu y tế mạng xã hội Việt Nam có con người kiểm duyệt nghiêm ngặt, minh chứng cho vai trò "Human-in-the-Loop".
3. **Hiện thực hóa Post-hoc Trust Engineering:** Triển khai Temperature Scaling giúp hạ thấp ECE tới 56%, đưa xác suất hiển thị về đúng giá trị tin cậy thực tế, đem lại độ an toàn cao khi ứng dụng vào thực tế y tế.
4. **Trực quan hóa XAI chuẩn khoa học:** Không dùng heuristic cứng, tích hợp Integrated Gradients để tính toán tầm quan trọng của từng từ (token-level attribution) giúp người dùng hiểu rõ quyết định của AI.

#### B. Hướng phát triển tương lai
* **Tối ưu hóa LLM sinh cấu trúc:** Áp dụng kỹ thuật sinh *Answer-First* (đưa kết quả nhãn lên trước, lập luận theo sau) để triệt tiêu tỷ lệ hỏng cú pháp (Parse Failure Rate 28.0%).
* **Mở rộng Corpus dữ liệu:** Gia tăng kích thước tập Gold Test và tích hợp dữ liệu đa ngôn ngữ khu vực Đông Nam Á.
* **Nhúng sâu vào hệ thống Y tế Công cộng:** Liên kết thử nghiệm với các trung tâm CDC địa phương làm bộ lọc thông tin sai lệch về vắc-xin tự động theo thời gian thực.

---

## 💻 HƯỚNG DẪN CÀI ĐẶT VÀ TRẢI NGHIỆM LOCAL (GETTING STARTED)

### 1. Cài đặt môi trường
Yêu cầu Python 3.10+ và các thư viện cốt lõi:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Ứng dụng Web (Streamlit/Gradio Demo)
Ứng dụng tích hợp bộ thu thập dữ liệu đa nguồn (Báo chí, YouTube, Facebook), công nghệ hiệu chuẩn độ tin cậy và giải thích Captum:
```bash
# Đối với ứng dụng Gradio App chính thức
python app_gradio/app.py
```

### 3. Khởi động Đường hầm Bảo mật Ngrok GPU (Dành cho Local Server)
Nếu anh tự host GPU và muốn kết nối với client trên HF Space bằng đường hầm bảo mật đã có chính sách chống Spam (Rate Limit 60 requests/phút):
Nhấp đúp trực tiếp vào file:
👉 **`Kich_Hoat_Duong_Ham.bat`** (Windows)
Hoặc chạy lệnh:
```bash
python run_ngrok_tunnel.py
```

---
*© 2026 VaccineNLP Project Team. Đồ án Tốt nghiệp xuất sắc HUPH 2026. Bản quyền thuộc về tác giả.*
