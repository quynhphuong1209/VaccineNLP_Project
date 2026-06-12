# 🗣️ Hệ Thống Thu Thập & Kỹ Thuật Dữ Liệu Lớn (src/data_pipeline/)

**Cập nhật:** 02/06/2026 · Phiên bản: 3.1 · Trạng thái: ✅ Được duy trì độc lập & Chuẩn hóa khoa học

Thư mục `src/data_pipeline/` đóng vai trò là phân hệ kỹ nghệ dữ liệu (Data Engineering Pipeline) của VaccineNLP. Đây là minh chứng kỹ thuật (Proof of Work) đại diện cho **Chương 3 (Phương pháp nghiên cứu - Giai đoạn 1 & 2)** của báo cáo luận văn tốt nghiệp, chịu trách nhiệm thu thập, lọc trùng xuyên phiên, ẩn danh hóa dữ liệu riêng tư và thống nhất cấu trúc dữ liệu từ nhiều nền tảng mạng xã hội khác nhau về phân tầng **Bronze** (thô).

---

## 🗂️ Cấu Trúc Chi Tiết Thư Mục (Granular File Index)

```
src/data_pipeline/
├── __init__.py
├── README.md                              ← Bạn đang ở đây
│
├── 📂 collection/                         # 📢 Phân hệ thu thập dữ liệu đa nguồn
│   ├── __init__.py
│   ├── master_collector_v2.py              # Động cơ điều phối chính (Orchestrator CLI v2.1)
│   ├── apify_social_collector_v2.py       # Tích hợp Apify API, xoay vòng token (v3.1)
│   ├── facebook_page_collector.py          # Kết nối Facebook Graph API trực tiếp
│   └── 📂 actor_configs/                  # Thư mục cấu hình JSON cho Apify Actors
│       ├── facebook_config.json            # Cấu hình cào fanpage/bình luận Facebook
│       ├── tiktok_config.json              # Cấu hình cào video/bình luận TikTok
│       ├── youtube_config.json             # Cấu hình cào video/caption YouTube
│       ├── threads_config.json             # Cấu hình cào Threads
│       └── rss_config.json                 # Cấu hình cào RSS báo chí
│
└── 📂 preprocessing/                      # 🔧 Module làm sạch thô (Legacy - Phiên bản nháp)
    ├── __init__.py
    ├── pipeline.py                        # Trình điều phối làm sạch thô
    ├── text_cleaner_v2.py                 # Bộ làm sạch HTML, URL thô
    ├── language_filter.py                 # Bộ lọc ngôn ngữ fasttext thô
    └── vn_tokenizer.py                    # Tokenizer phân tách từ
```

---

## 📢 Động Cơ Thu Thập Đa Nguồn (Collection Engines)

Hệ thống được thiết kế theo cấu trúc mô-đun hóa cao độ để cào dữ liệu từ 5 nguồn truyền thông số phổ biến tại Việt Nam:

### 1️⃣ **master_collector_v2.py** (Orchestrator CLI v2.1)
Điều phối các kịch bản cào quét song song giữa các nền tảng thông qua các tham số dòng lệnh cực kỳ linh hoạt:
- **Nguyên lý khử trùng trùng lặp xuyên phiên (Cross-session Deduplication):**
  - Sử dụng đối tượng `DedupStore` để ghi nhận các URL và nội dung văn bản đã cào trong quá khứ.
  - Tự động sinh `global_id` duy nhất cho mỗi bài viết theo cú pháp: `{source}_{post_id}`, trong đó `post_id` được băm bằng MD5 của 80 ký tự đầu của nội dung bài viết nếu nguồn thô thiếu ID chính thức.
  - Khi phát hiện trùng lặp, kịch bản sẽ thực hiện so sánh và ưu tiên giữ lại phiên bản có `data_confidence_score` cao hơn.
- **Source Distribution Reporting:** Tự động kết xuất biểu đồ thanh ASCII mô tả phân bổ phần trăm dữ liệu theo nguồn thô ngay sau khi kết thúc phiên cào.

---

### 2️⃣ **apify_social_collector_v2.py** (Bộ Tích Hợp Apify v3.1)
Module lõi kết nối trực tiếp với dịch vụ đám mây Apify để vượt qua các cơ chế tường lửa và chặn IP (Anti-scraping) của các ông lớn mạng xã hội:
- **Chiến lược cào Facebook 2-Actor (Routing thông minh):**
  - Quét danh sách URL đầu vào để định tuyến: Các URL bài viết nhóm (`/groups/`) được chuyển cho actor chuyên biệt `apify/facebook-groups-scraper`. Các URL bài viết trang cá nhân hoặc fanpage (`/posts/`) được chuyển cho actor `apify/facebook-comments-scraper` với cấu hình đệ quy `RANKED_THREADED` để lấy tối đa các reply phụ.
- **Chiến lược cào TikTok 2-Phase (Deep Extraction):**
  - **Phase 1 (Discovery):** Gọi actor `clockworks/free-tiktok-scraper` tìm kiếm hàng loạt các video IDs dựa trên 25 hashtag chia làm 6 nhóm (General, Covid, Side-effects, Children, Specific, Stance).
  - **Phase 2 (Deep comments):** Đẩy danh sách video IDs thu được ở Phase 1 vào actor `clockworks/tiktok-scraper` để thu hoạch sâu toàn bộ bình luận của người dùng cuối.
- **Xoay tua token tự động (Token Rotation):**
  - Tự động phát hiện lỗi hạn ngạch (Rate limit) hoặc token hết hạn. Khi xảy ra lỗi, hệ thống sẽ thực hiện chuyển đổi xoay vòng tự động từ `APIFY_API_TOKEN_1` đến `APIFY_API_TOKEN_5` cấu hình trong `.env` chỉ trong **50ms** để đảm bảo phiên cào không bị gián đoạn.
- **Từ lóng và mật ngữ y tế (Slang Discovery Keywords):**
  - Tích hợp các từ lóng tiêm do dự nhằm tối đa hóa khả năng cào trúng tin giả y khoa tại Việt Nam:
    - *Trải nghiệm tiêu cực:* `kiếp nạn`, `bị hành`, `vật lên vật xuống`, `sốt li bì`, `co giật`.
    - *Trái chiều:* `chê nha`, `phốt tiêm chủng`, `lùa gà`, `tiền mất tật mang`.
    - *Thuận tự nhiên:* `để tự nhiên`, `thuận tự nhiên`, `chữa lành`, `miễn dịch tự nhiên`, `bài thuốc dân gia`.

---

## 🏃 Quy Trình Vận Hành & Lệnh Chạy (Execution Commands)

Để khởi chạy kỹ nghệ thu thập dữ liệu đa nguồn từ Command Line:

### Khởi chạy cào quét toàn bộ nền tảng (Bật tất cả cấu hình)
```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Chạy Master Collector cào song song VnExpress, Tuổi Trẻ, Reddit, YouTube và Apify
python src/data_pipeline/collection/master_collector_v2.py --all --max-articles 30 --max-social 100
```

### Chỉ cào quét mạng xã hội Facebook & TikTok qua Apify
```bash
python src/data_pipeline/collection/master_collector_v2.py --with-apify --skip-news --max-social 200
```

### Chỉ cào quét video và phụ đề y tế công cộng YouTube (Không cần API key)
```bash
python src/data_pipeline/collection/master_collector_v2.py --with-youtube --skip-news
```

---

## 🔐 Bảo Vệ Thông Tin Riêng Tư & Ẩn Danh Hóa (PII Scrubbing)

Tuân thủ nghiêm ngặt tiêu chuẩn đạo đức nghiên cứu y tế công cộng của đồ án. Trước khi lưu dữ liệu vào đĩa cứng, kịch bản thực hiện ẩn danh hóa thông tin định danh cá nhân (PII) tự động:
- **Số điện thoại:** Dò quét bằng biểu thức chính quy (Regex) `(\+84|0)[0-9]{8,10}` đặt lại thành `[SĐT ẨN]`.
- **Đường dẫn cá nhân:** Tìm các URL facebook.com/profile hoặc tiktok.com/profile đặt lại thành `[PROFILE ẨN]`.
- **Tên tài khoản người dùng:** Sử dụng thuật toán băm SHA-256 một chiều để mã hóa các chuỗi `@username` thành mã định danh không thể truy ngược:
  $$\text{Username}_{\text{anonymous}} = \text{SHA-256}(\text{Username}_{\text{raw}})[0:16]$$

---

## 📊 Định Dạng Dữ Liệu Bronze Chuẩn Hóa (JSON Schema)

Dữ liệu thô sau thu thập được chuẩn hóa về một cấu trúc JSON thống nhất lưu tại `datasets/01_raw/combined_vaccine_[timestamp].json`:

```json
{
  "source": "facebook",
  "platform": "facebook",
  "source_credibility_tier": "non-institutional",
  "post_id": "8f3e9b2a1c0d4e5f",
  "text": "Tôi nghe nói tiêm chủng mở rộng mũi 5 trong 1 làm trẻ bị sốt cao hành [SĐT ẨN] @8c7b6f5a4d3e2b1a",
  "url": "https://www.facebook.com/posts/123456789",
  "engagement_metrics": {
    "likes": 120,
    "comments": 45,
    "views": 0,
    "shares": 12
  },
  "timestamp": "2026-05-20T10:15:30Z",
  "collected_at": "2026-06-02T15:00:00Z",
  "collection_method": "apify-facebook-comments-scraper",
  "data_confidence_score": 0.75,
  "language": "vi"
}
```

---

## 🔧 Phân Hệ Tiền Xử Lý Cũ (Legacy `preprocessing/`)

- **Tại sao phân hệ này vẫn tồn tại?**
  - Đây là phiên bản nháp đầu tiên (Prototype) của pipeline tiền xử lý được phát triển trong Giai đoạn 2 của đồ án. 
  - Nó được giữ lại trong thư mục `src/data_pipeline/preprocessing/` để phục vụ mục đích đối chiếu lịch sử phát triển code (Audit trail) và phục vụ các bài toán thử nghiệm độc lập không phụ thuộc vào thư viện PyTorch nặng nề của core pipeline.
- **Sự khác biệt:** Phân hệ cũ chỉ áp dụng làm sạch chuỗi thô cơ bản, lọc tiếng Việt đơn giản và không có các tính năng tối ưu cho huấn luyện của core pipeline `src/preprocessing/` như: lọc veterinary human context y tế, băm stable post_id chống xung đột, hay công cụ `corpus_audit()` phục vụ dán nhãn thủ công.

---

## 🔍 Xử Lý Lỗi Thường Gặp Khi Thu Thập (Troubleshooting)

| Sự Cố | Nguyên Nhân | Giải Pháp |
|---|---|---|
| **Lỗi 401 Unauthorized** | APIFY_API_TOKEN cấu hình sai hoặc hết hạn. | Kiểm tra tệp `.env`, cập nhật token mới hoặc cấu hình thêm `APIFY_API_TOKEN_BACKUP` để kích hoạt xoay tua. |
| **Số mẫu thu thập bằng 0** | Cơ chế chặn Scraping của Facebook kích hoạt (Token Guard). | Apify Scraper sẽ tự động chuyển đổi proxy sang Residential IP, hoặc thuyết trình viên giảm kích thước batch xuống còn 1 post/lần gọi. |
| **Không import được collector** | Lỗi import tương đối do sai thư mục làm việc. | Script tự động gọi `ensure_src_in_sys_path()` để đưa `src/` vào sys.path trước khi chạy. |

---

*VaccineNLP Data Engineering Infrastructure Team · HUPH 2026*
