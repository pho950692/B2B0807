# core_app.py
import os
from flask import Flask, send_from_directory, render_template
import pytesseract

ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "templates"),
    static_folder=os.path.join(ROOT, "static"),
)

# 把可用的 endpoint 名稱清單提供給所有模板
@app.context_processor
def inject_endpoints():
    # 例如：{'home', 'search', 'manual_invoice', ...}
    return {
        "endpoints": set(app.view_functions.keys()),
        "has_endpoint": lambda ep: ep in app.view_functions,  # 可用在 if 判斷
    }

app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# core_app.py
app.config.update(
    ROOT_DIR=ROOT,
    UPLOAD_FOLDER=os.path.join(ROOT, "uploads"),
    CROPPED_FOLDER=os.path.join(ROOT, "uploads", "cropped"),
    POPPLER_PATH=os.environ.get("POPPLER_PATH", r"C:\Users\user\Downloads\Release-24.08.0-0 (1)\poppler-24.08.0\Library\bin"),  # ← 修正拼字
    TESSERACT_CMD=os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
)

# 讓 ocr_utils 可以用 os.environ 讀到
os.environ["POPPLER_PATH"] = app.config["POPPLER_PATH"]       

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CROPPED_FOLDER"], exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = app.config["TESSERACT_CMD"]
tess_dir = os.path.dirname(app.config["TESSERACT_CMD"])
os.environ["TESSDATA_PREFIX"] = os.path.join(tess_dir, "tessdata")

print("[CORE] ROOT_DIR         =", app.config["ROOT_DIR"])
print("[CORE] UPLOAD_FOLDER    =", app.config["UPLOAD_FOLDER"])
print("[CORE] CROPPED_FOLDER   =", app.config["CROPPED_FOLDER"])
print("[CORE] POPPLER_PATH     =", app.config["POPPLER_PATH"])
print("[CORE] TESSERACT_CMD    =", pytesseract.pytesseract.tesseract_cmd)
print("[CORE] TESSDATA_PREFIX  =", os.environ.get("TESSDATA_PREFIX", ""))

@app.route("/")
def home():
    return render_template("home.html")

# ✅ 唯一版本（yr.py 不再定義同路徑）
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/uploads/cropped/<path:filename>")
def uploads_cropped(filename):
    return send_from_directory(app.config["CROPPED_FOLDER"], filename)