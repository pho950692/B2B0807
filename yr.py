# yr.py
# -*- coding: utf-8 -*-
import os, uuid, sys, glob
from typing import Dict, Any, List
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import request, jsonify, render_template, url_for, flash, send_from_directory
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

VENDOR_NAME_MAP = {
    'mi': 'Microsoft',
    'op': 'OpenAI',
    'pc': 'PChome',
}

def _enrich_vendor_name(one: dict):
    # 可能有的鍵名：label / cls / type，依你的回傳結構調整
    label = (one.get('label') or one.get('cls') or one.get('type') or '').lower()
    if not one.get('name') and label in VENDOR_NAME_MAP:
        one['name'] = VENDOR_NAME_MAP[label]
    return one

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
    # 取得最近一次辨識結果
    results = LAST_RESULTS if LAST_RESULTS else []  # 依你專案模板名稱（你之前是 auto_inv.html / 或 invoice_auto.html）
    return render_template("auto_inv.html", results=results)

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
            conf = info.get("conf", {})  # YOLO信心分數 dict
            row = {
                "origin":   raw,
                "filename": out_name,
                "imageUrl": url_for("uploads", filename=out_name),
                "type":     info.get("type", ""),
                "num":      info.get("num", ""),
                "sun":      info.get("sun", ""),
                "date":     info.get("date", ""),
                "cash":     info.get("cash", ""),
                "score":    conf,  # 直接回傳 YOLO信心分數 dict
                "bnu":      "",
                "name":     VENDOR_NAME_MAP.get((info.get("type") or "").lower(), ""),
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

# === 相機 ===
@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route("/upload_camera", methods=["POST"])
def upload_camera():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "沒有收到檔案"}), 400

    raw = secure_filename(f.filename or f"camera_{uuid.uuid4().hex}.jpg")
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], raw)
    f.save(out_path)

    # 直接跑 YOLO + OCR
    info = detect_and_ocr(out_path, crops_dir=app.config["CROPPED_FOLDER"])
    row = {
        "filename": raw,
        "imageUrl": url_for("uploads", filename=raw),
        "num":  info.get("num",""),
        "sun":  info.get("sun",""),
        "date": info.get("date",""),
        "cash": info.get("cash",""),
        "bnu":  "",
        "name": VENDOR_NAME_MAP.get((info.get("type") or "").lower(), ""),
        "add":  ""
    }
    return jsonify([row])



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
# yr.py
@app.route("/result/detail/<path:imgname>")
def result_detail(imgname):
    ocr = detect_and_ocr(str(UPLOAD_DIR / imgname), crops_dir=str(CROPS_DIR))

    # YOLO 已回傳 {"key":..., "path": 檔名}；模板會用 url_for('uploads_cropped', filename=item.cropped_image)
    crop_by_key = {c["key"]: c.get("path") for c in ocr.get("crops", [])}

    texts = [
        {"label": "發票號碼", "text": ocr.get("num",""),  "key":"num",  "cropped_image": crop_by_key.get("num","")},
        {"label": "統一編號", "text": ocr.get("sun",""),  "key":"sun",  "cropped_image": crop_by_key.get("sun","")},
        {"label": "日期",     "text": ocr.get("date",""), "key":"date", "cropped_image": crop_by_key.get("date","")},
        {"label": "價格",     "text": ocr.get("cash",""), "key":"cash", "cropped_image": crop_by_key.get("cash","")},
        {"label": "買方編號", "text": "", "key":"bnu"},
        {"label": "公司名稱", "text": VENDOR_NAME_MAP.get((ocr.get("type") or "").lower(), ""), "key":"name"},
        {"label": "公司地址", "text": "", "key":"add"},
    ]

    row = {
        "filename": imgname,
        "imageUrl": url_for("uploads", filename=imgname)
    }
    return render_template("result.html", result=row, texts=texts)

# === 最近一次辨識結果 API ===
@app.route('/progress/last', methods=['GET'])
def progress_last():
    return jsonify({"results": LAST_RESULTS})