# yr.py
# -*- coding: utf-8 -*-
import os, uuid, sys, glob
from typing import Dict, Any, List
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import request, jsonify, render_template, url_for, flash
from core_app import app  # 只使用 core_app 的 app

# 路徑
BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CROPS_DIR  = UPLOAD_DIR / "cropped"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# 供匯入
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# YOLO / OCR
try:
    from yocr.yolo import detect_and_ocr
    from yocr.ocr_utils import pdf_to_images
except ModuleNotFoundError:
    from yolo import detect_and_ocr
    from ocr_utils import pdf_to_images

# === 內部工具 ===
PROGRESS: Dict[str, Dict[str, Any]] = {}
LAST_RESULTS: List[Dict[str, Any]] = []

def _progress_start(job_id: str, total: int):
    PROGRESS[job_id] = {"status": "running", "total": total, "done": 0, "error": "", "finished": False}

def _progress_step(job_id: str):
    d = PROGRESS.get(job_id)
    if d: d["done"] = int(d.get("done", 0)) + 1

def _progress_finish(job_id: str, error: str = ""):
    d = PROGRESS.get(job_id)
    if not d: return
    d["status"] = "error" if error else "ok"
    d["error"]  = error
    d["finished"] = True

def _find_crop(filename: str, key: str) -> str:
    base = os.path.splitext(filename)[0]
    pattern = str(CROPS_DIR / f"{base}_*_{key}.jpg")  # 支援 nonce
    matches = glob.glob(pattern)
    return os.path.basename(matches[0]) if matches else ""

# === 首頁 ===
@app.route("/invoice/auto", methods=["GET"], endpoint="invoice_auto")
def yr_home():
    # 依你專案模板名稱（你之前是 auto_inv.html / 或 invoice_auto.html）
    return render_template("auto_inv.html")

# === 上傳與辨識 ===
@app.route("/upload", methods=["POST"], endpoint="yr_upload")
def yr_upload():
    job_id = (request.form.get("job_id") or request.values.get("job_id") or uuid.uuid4().hex)
    files = request.files.getlist("files")
    if not files:
        flash("請選擇檔案再上傳")
        return jsonify({"error": "沒有選擇檔案"}), 400

    _progress_start(job_id, total=len(files))
    results: List[Dict[str, Any]] = []

    try:
        for f in files:
            raw = secure_filename(f.filename or f"img_{uuid.uuid4().hex}.jpg")
            base, ext = os.path.splitext(raw)

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

            info = detect_and_ocr(out_path, crops_dir=str(CROPS_DIR))

            # ✅ 用 core_app 的 uploads endpoint
            img_url = url_for("uploads", filename=out_name)
            row = {
                "origin":   raw,
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
        _progress_finish(job_id)  # 這行會把 finished 設 True
        first_url = url_for("yr_result", filename=results[0]["filename"]) if results else url_for("invoice_auto")
        return jsonify({"results": results, "job_id": job_id, "open": "results", "first": first_url})
    
    except Exception as e:
        _progress_finish(job_id, str(e))
        return jsonify({"error": str(e), "job_id": job_id}), 500

# === 進度查詢 ===
@app.route("/progress/<job_id>", methods=["GET"], endpoint="yr_progress")
def yr_progress(job_id: str):
    return jsonify(PROGRESS.get(job_id) or {"status": "missing", "total": 0, "done": 0, "error": "", "finished": True})

# === 結果頁（推裁切圖）===
@app.route("/result/<path:filename>", methods=["GET"], endpoint="yr_result")
def yr_result(filename: str):
    row = next((x for x in LAST_RESULTS if x.get("filename") == filename), None)
    if row is None:
        row = {
            "origin": filename, "filename": filename,
            "imageUrl": url_for("uploads", filename=filename),
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

# === 即時重跑一張（保證最新）===
@app.route("/result/detail/<path:imgname>")
def result_detail(imgname):
    ocr = detect_and_ocr(str(UPLOAD_DIR / imgname), crops_dir=str(CROPS_DIR))
    # ✅ 用 core_app 的 uploads_cropped endpoint
    crops_map = {c["key"]: url_for("uploads_cropped", filename=c["path"]) for c in ocr.get("crops", [])}
    return render_template("result.html", data=ocr, crops=crops_map)