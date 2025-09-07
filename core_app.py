# core_app.py
import os
from flask import Flask, render_template, send_from_directory
import pytesseract

ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "templates"),
    static_folder=os.path.join(ROOT, "static"),
)

# ===== Secret =====
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# ===== 路徑 & 權重設定（依你原本結構延伸成二階段） =====
app.config.update(
    ROOT_DIR=ROOT,
    UPLOAD_FOLDER=os.path.join(ROOT, "uploads"),
    CROPPED_FOLDER=os.path.join(ROOT, "uploads", "cropped"),

    # ▶ 二階段：先用 router 判定 pc/op/mi，再載入欄位模型
    ROUTER_WEIGHTS=os.environ.get("ROUTER_WEIGHTS", r"C:\Users\user\Downloads\tr3.pt"),
    PC_WEIGHTS    =os.environ.get("PC_WEIGHTS",     r"C:\Users\user\Downloads\pc.pt"),
    OP_WEIGHTS    =os.environ.get("OP_WEIGHTS",     r"C:\Users\user\Downloads\op.pt"),
    MI_WEIGHTS    =os.environ.get("MI_WEIGHTS",     r"C:\Users\user\Downloads\mi.pt"),

    POPPLER_PATH=os.environ.get("POPPLER_PATH", r"D:\poppler\Library\bin"),
    TESSERACT_CMD=os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
)

# 確保上傳資料夾存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CROPPED_FOLDER"], exist_ok=True)

# Windows：指定 tesseract & tessdata
pytesseract.pytesseract.tesseract_cmd = app.config["TESSERACT_CMD"]
tessdata_dir = os.path.join(os.path.dirname(app.config["TESSERACT_CMD"]), "tessdata")
os.environ["TESSDATA_PREFIX"] = tessdata_dir

# ===== 基本頁 =====
@app.route("/")
def home():
    # 你原有的首頁（保留 endpoint 名稱 'home'）
    return render_template("home.html")

# 與 auto_inv.html 連結一致：回到上傳頁
@app.route("/invoice/auto")
def invoice_auto():
    return render_template("auto_inv.html")

# ===== 靜態檔案（原始與裁切）=====
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# 舊名保留
@app.route("/uploads/cropped/<path:filename>")
def uploads_cropped(filename):
    return send_from_directory(app.config["CROPPED_FOLDER"], filename)


# 新名：配合 result.html 使用 url_for('uploaded_cropped_file', ...)
@app.route("/uploaded/cropped/<path:filename>")
def uploaded_cropped_file(filename):
    return send_from_directory(app.config["CROPPED_FOLDER"], filename)
