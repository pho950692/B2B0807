# -*- coding: utf-8 -*-
"""
Flask routes for invoice system (fixed)
- 修正：重複 endpoint 'yr_uploads_cropped'
- 修正：不再覆蓋 core_app.app
- 裁切圖查找支援帶隨機碼：{base}_<nonce>_{key}.jpg
"""
import os, uuid, sys, glob
from typing import Dict, Any, List
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import (
    request, jsonify, render_template, url_for, send_from_directory
)
from core_app import app  # 只使用 core_app 提供的 app，避免被覆蓋  (關鍵修正)

# === 路徑設定 ===
BASE_DIR  = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CROPS_DIR  = UPLOAD_DIR / "cropped"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# 確保本目錄在 sys.path 中（供 from yolo / ocr_utils 匯入）
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 匯入 YOLO / OCR（優先 yocr/，否則使用本地模組）
try:
    from yocr.yolo import detect_and_ocr
    from yocr.ocr_utils import pdf_to_images
except ModuleNotFoundError:
    from yolo import detect_and_ocr             # 你目前專案的 yolo.py
    from ocr_utils import pdf_to_images         # 你目前專案的 ocr_utils.py

# === 靜態檔案服務 ===
@app.route("/uploads/<path:filename>", endpoint="yr_uploads")
def yr_uploads(filename: str):
    return send_from_directory(str(UPLOAD_DIR), filename, as_attachment=False)

@app.route("/uploads/cropped/<path:filename>", endpoint="yr_uploads_cropped")
def yr_uploads_cropped(filename: str):
    return send_from_directory(str(CROPS_DIR), filename, as_attachment=False)

# ⚠️ 千萬不要再定義第二個同名 endpoint (yr_uploads_cropped)！
# 你原檔案同時又定義了 /yr/uploads/cropped/... 並且 endpoint 也叫 yr_uploads_cropped，會引發 AssertionError。已移除。  # noqa

# === 進度/快取 ===
PROGRESS: Dict[str, Dict[str, Any]] = {}
LAST_RESULTS: List[Dict[str, Any]] = []

def _progress_start(job_id: str, total: int):
    PROGRESS[job_id] = {"status": "running", "total": total, "done": 0, "error": ""}

def _progress_step(job_id: str):
    d = PROGRESS.get(job_id)
    if d: d["done"] = int(d.get("done", 0)) + 1

def _progress_finish(job_id: str, error: str = ""):
    d = PROGRESS.get(job_id)
    if not d: return
    if error:
        d["status"] = "error"; d["error"] = error
    else:
        d["status"] = "ok"

# === 裁切圖查找（支援隨機碼）===
def _find_crop(filename: str, key: str) -> str:
    """
    回傳對應欄位的小圖檔名（存在才回傳），交給模板用 url_for('yr_uploads_cropped', filename=...)。
    以前只找 {base}_{key}.jpg，現在支援 {base}_*_{key}.jpg（帶 nonce）。
    """
    base = os.path.splitext(filename)[0]
    pattern = str(CROPS_DIR / f"{base}_*_{key}.jpg")
    matches = glob.glob(pattern)
    if matches:
        # 只回傳檔名，模板再用 url_for('yr_uploads_cropped', filename=...) 組 URL
        return os.path.basename(matches[0])
    return ""

# === 首頁 ===
@app.route("/invoice/auto", methods=["GET"], endpoint="yr_home")
def yr_home():
    # 依你專案的模板命名；你現有檔案是 auto_inv.html
    return render_template("auto_inv.html")

# === 上傳與辨識 ===
@app.route("/upload", methods=["POST"], endpoint="yr_upload")
def yr_upload():
    job_id = (request.form.get("job_id") or request.values.get("job_id") or uuid.uuid4().hex)
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "沒有選擇檔案"}), 400

    _progress_start(job_id, total=len(files))
    results: List[Dict[str, Any]] = []

    try:
        for f in files:
            raw_name = secure_filename(f.filename or f"img_{uuid.uuid4().hex}.jpg")
            base, ext = os.path.splitext(raw_name)

            # PDF 轉圖（取第一頁）
            if ext.lower() == ".pdf":
                tmp_pdf = str(UPLOAD_DIR / f"{base}_{uuid.uuid4().hex}.pdf")
                f.save(tmp_pdf)
                pil_imgs = pdf_to_images(tmp_pdf)
                if not pil_imgs:
                    raise RuntimeError("PDF 轉圖失敗")
                out_name = f"{base}_{uuid.uuid4().hex}.jpg"
                out_path = str(UPLOAD_DIR / out_name)
                pil_imgs[0].save(out_path, "JPEG", quality=95)
            else:
                out_name = f"{base}_{uuid.uuid4().hex}{ext or '.jpg'}"
                out_path = str(UPLOAD_DIR / out_name)
                f.save(out_path)

            info = detect_and_ocr(out_path, crops_dir=str(CROPS_DIR))  # 產出 fields + crops

            img_url = url_for("yr_uploads", filename=out_name)
            row = {
                "origin":   raw_name,
                "filename": out_name,
                "imageUrl": img_url,
                "type":     info.get("type", ""),
                "num":      info.get("num", ""),
                "sun":      info.get("sun", ""),
                "date":     info.get("date", ""),
                "cash":     info.get("cash", ""),
                "bnu":      "",
                "name":     "",
                "add":      "",
            }
            results.append(row)
            _progress_step(job_id)

        LAST_RESULTS.clear()
        LAST_RESULTS.extend(results)
        _progress_finish(job_id)
        return jsonify({"results": results, "job_id": job_id, "open": "results"})

    except Exception as e:
        _progress_finish(job_id, str(e))
        return jsonify({"error": str(e), "job_id": job_id}), 500

# === 進度查詢 ===
@app.route("/progress/<job_id>", methods=["GET"], endpoint="yr_progress")
def yr_progress(job_id: str):
    return jsonify(PROGRESS.get(job_id) or {"status": "missing", "total": 0, "done": 0, "error": ""})

# === 單張詳細頁（使用上次結果 + 檔名推裁切圖）===
@app.route("/result/<path:filename>", methods=["GET"], endpoint="yr_result")
def yr_result(filename: str):
    row = None
    for x in LAST_RESULTS:
        if x.get("filename") == filename:
            row = x
            break

    if row is None:
        img_url = url_for("yr_uploads", filename=filename)
        row = {
            "origin": filename, "filename": filename, "imageUrl": img_url,
            "type": "", "num": "", "sun": "", "date": "", "cash": "",
            "bnu": "", "name": "", "add": ""
        }

    texts = [
        {"label": "發票號碼", "text": row.get("num",""),  "key":"num",  "cropped_image": _find_crop(row["filename"], "num")},
        {"label": "統一編號", "text": row.get("sun",""),  "key":"sun",  "cropped_image": _find_crop(row["filename"], "sun")},
        {"label": "日期",     "text": row.get("date",""), "key":"date", "cropped_image": _find_crop(row["filename"], "date")},
        {"label": "價格",     "text": row.get("cash",""), "key":"cash", "cropped_image": _find_crop(row["filename"], "cash")},
        {"label": "買方編號", "text": row.get("bnu",""),  "key":"bnu"},
        {"label": "公司名稱", "text": row.get("name",""), "key":"name"},
        {"label": "公司地址", "text": row.get("add",""),  "key":"add"},
    ]

    return render_template("result.html", result=row, texts=texts)

# === 另一種詳細頁（即時重跑 YOLO+OCR；保證最新）===
@app.route("/result/<path:imgname>")
def result_detail(imgname):
    img_path = UPLOAD_DIR / imgname
    ocr = detect_and_ocr(str(img_path), crops_dir=str(CROPS_DIR))
    crops_map = {c["key"]: url_for("yr_uploads_cropped", filename=c["path"]) for c in ocr.get("crops", [])}
    return render_template("result.html", data=ocr, crops=crops_map)

# === 直接啟動（開發用）===
if __name__ == "__main__":
    app.run(debug=True)
