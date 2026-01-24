## API Endpoints
baseURL=http://minerudeployment-production.up.railway.app/
### 1. Parse PDF (Async) - `/parse/`

**Method**: `POST`

**Purpose**: Submit a PDF for asynchronous processing

**Request**:
```bash
curl -X POST "{baseURL}/parse/" \
  -F "file=@document.pdf" \
  -F "parse_formula=true" \
  -F "parse_table=true" \
  -F "parse_ocr=true" \
  -F "extract_images=true" \
  -F "output_format=both" \
  -F "dpi=200"
```

**Parameters**:
- `file` (required): PDF file (max 100MB)
- `parse_formula` (optional, default: true): Enable formula recognition
- `parse_table` (optional, default: true): Enable table extraction
- `parse_ocr` (optional, default: true): Enable OCR for scanned documents
- `extract_images` (optional, default: true): Extract embedded images
- `output_format` (optional, default: "both"): "markdown", "json", or "both"
- `dpi` (optional, default: 200): Image extraction DPI
- `force_gpu` (optional, default: false): Force GPU processing

**Response**:
```json
{
  "task_id": "task_1732345678900",
  "status": "pending",
  "message": "PDF parsing started using modal_gpu",
  "backend": "modal_gpu",
  "estimated_time": 60
}
```

**Status Codes**:
- `200 OK`: Task created successfully
- `400 Bad Request`: Invalid file or parameters
- `413 Payload Too Large`: File exceeds 100MB
- `500 Internal Server Error`: Server error

---

### 2. Get Task Status - `/status/{task_id}`

**Method**: `GET`

**Purpose**: Check processing status and retrieve results

**Request**:
```bash
curl "{baseURL}/status/task_1732345678900"
```

**Response (Pending)**:
```json
{
  "task_info": {
    "task_id": "task_1732345678900",
    "status": "pending",
    "backend": "modal_gpu",
    "created_at": 1732345678.9,
    "started_at": null,
    "completed_at": null,
    "error_message": null,
    "file_info": {
      "filename": "document.pdf",
      "size": 1024000,
      "md5": "abc123..."
    }
  },
  "content": null,
  "download_urls": null
}
```

**Response (Processing)**:
```json
{
  "task_info": {
    "task_id": "task_1732345678900",
    "status": "processing",
    "backend": "modal_gpu",
    "created_at": 1732345678.9,
    "started_at": 1732345680.5,
    "completed_at": null,
    "error_message": null,
    "file_info": {...}
  },
  "content": null,
  "download_urls": null
}
```

**Response (Completed)**:
```json
{
  "task_info": {
    "task_id": "task_1732345678900",
    "status": "completed",
    "backend": "modal_gpu",
    "created_at": 1732345678.9,
    "started_at": 1732345680.5,
    "completed_at": 1732345720.3,
    "error_message": null,
    "file_info": {
      "filename": "document.pdf",
      "size": 1024000,
      "md5": "abc123..."
    }
  },
  "content": {
    "markdown_content": "# Document Title\n\nContent...",
    "json_content": "# Metadata returned from MinerU, need to be stored into a json file",
    "images": ["base64_image_1", "base64_image_2"],
    "page_count": 10,
    "processing_time": 39.8,
    "metadata": {
      "backend": "modal_gpu",
      "status": "completed"
    }
  },
  "download_urls": {
    "zip": "https://storage.example.com/abc123.../results.zip"
  }
}
```

**Task Status Values**:
- `pending`: Task created, waiting to start
- `processing`: Currently being processed
- `completed`: Processing finished successfully
- `failed`: Processing failed (check error_message)
- `cached`: Result retrieved from cache (shown as completed to clients)

**Status Codes**:
- `200 OK`: Task found
- `404 Not Found`: Task ID doesn't exist