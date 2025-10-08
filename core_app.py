
# core_app.py

import os
from flask import Flask, send_from_directory, render_template
import pytesseract
import sys
import importlib.util

# 從 config.py 讀固定設定（每台電腦各自修改 config.py）
from config import TESSERACT_CMD, POPPLER_PATH

# 專案根目錄
ROOT = os.path.dirname(os.path.abspath(__file__))

# 建立 Flask
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "templates"),
    static_folder=os.path.join(ROOT, "static"),
)

# Secret Key
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# === 路徑設定（只用 config.py 的值）===
app.config.update(
    ROOT_DIR=ROOT,
    UPLOAD_FOLDER=os.path.join(ROOT, "uploads"),
    CROPPED_FOLDER=os.path.join(ROOT, "uploads", "cropped"),
    POPPLER_PATH=POPPLER_PATH,      # 由 config.py 指定
    TESSERACT_CMD=TESSERACT_CMD,    # 由 config.py 指定
)

# 確保上傳資料夾存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CROPPED_FOLDER"], exist_ok=True)

# === Tesseract 設定（Windows）===
# 指定 tesseract.exe 路徑
pytesseract.pytesseract.tesseract_cmd = app.config["TESSERACT_CMD"]
# 指定 TESSDATA_PREFIX（同資料夾下的 tessdata）
tess_dir = os.path.dirname(app.config["TESSERACT_CMD"])
os.environ["TESSDATA_PREFIX"] = os.path.join(tess_dir, "tessdata")


# === 將可用的 endpoint 名稱提供給所有模板（避免 current_app 未注入問題）===
@app.context_processor
def inject_endpoints():
    return {
        "endpoints": set(app.view_functions.keys()),
        "has_endpoint": lambda ep: ep in app.view_functions,
    }

# === 自動抓取 Email 發票附件（每分鐘） ===
def run_email_fetcher():
    try:
        spec = importlib.util.spec_from_file_location("email_invoice_fetcher", os.path.join(app.config["ROOT_DIR"], "email_invoice_fetcher.py"))
        email_fetcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(email_fetcher)
        email_fetcher.fetch_invoices()
        print("[APScheduler] 已自動執行 fetch_invoices()")
    except Exception as e:
        print(f"[APScheduler] 執行 fetch_invoices() 發生錯誤: {e}")

if __name__ != "__main__":
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(run_email_fetcher, 'interval', minutes=1, id='email_fetcher_job', replace_existing=True)
        scheduler.start()
    except Exception as e:
        print(f"[APScheduler] 啟動失敗: {e}")

# 啟動時印出確認資訊（方便你與組員檢查實際路徑）
print("[CORE] ROOT_DIR         =", app.config["ROOT_DIR"])
print("[CORE] UPLOAD_FOLDER    =", app.config["UPLOAD_FOLDER"])
print("[CORE] CROPPED_FOLDER   =", app.config["CROPPED_FOLDER"])
print("[CORE] POPPLER_PATH     =", app.config["POPPLER_PATH"])
print("[CORE] TESSERACT_CMD    =", pytesseract.pytesseract.tesseract_cmd)
print("[CORE] TESSDATA_PREFIX  =", os.environ.get("TESSDATA_PREFIX", ""))

# === 首頁 ===
@app.route("/")
def home():
    return render_template("home.html")

# === 靜態檔案：上傳原圖（唯一版本）===
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# === 靜態檔案：裁切圖（唯一版本）===
@app.route("/uploads/cropped/<path:filename>")
def uploads_cropped(filename):
    return send_from_directory(app.config["CROPPED_FOLDER"], filename)

# --- 健康檢查端點（貼到 core_app.py 最底部） ---
@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/db_ping")
def db_ping():
    from db import get_db
    try:
        conn = get_db()
        if conn is None:
            return ("db_fail: connect None", 500)
        cur = conn.cursor()
        cur.execute("SELECT NOW()")
        now_ts = cur.fetchone()
        cur.close(); conn.close()
        return f"db_ok: {now_ts}", 200
    except Exception as e:
        return ("db_fail: " + str(e), 500)