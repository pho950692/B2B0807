from flask import (
    Flask, request, render_template, redirect, url_for,
    flash, session, jsonify, send_from_directory
)
import os
import re
import uuid
import cv2
import torch
import pytesseract
import mysql.connector
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pdf2image import convert_from_path
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from config import db_config
from glob import glob

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 建議改為環境變數

# ---------- 基本設定 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PDF 轉圖工具（你機器上的 poppler bin）
POPPLER_PATH = os.path.join(BASE_DIR, 'poppler', 'Library', 'bin')

# 上傳與裁切資料夾
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CROPPED_FOLDER = os.path.join(UPLOAD_FOLDER, 'cropped')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

# Tesseract 執行檔路徑（依你電腦調整）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# YOLO 權重（依你實際檔案）
weight_path = r'C:\b2b0806\b2b_st\weights\best.pt'
if not os.path.exists(weight_path):
    raise FileNotFoundError(f"❌ 找不到模型權重檔案：{weight_path}")

model = torch.hub.load('ultralytics/yolov5:v6.2', 'custom',
                       path=weight_path, force_reload=True)

# 測試
print("CROPPED_FOLDER =", CROPPED_FOLDER)
print("Some crops:", os.listdir(CROPPED_FOLDER)[:10])


# 批次進度
PROGRESS = defaultdict(lambda: {
    "total": 0,
    "done": 0,
    "started_at": None,
    "finished": False,
    "error": ""
})

# 記憶體快取：單張結果頁使用
processed_results = {}


# ---------- 共用工具 ----------
def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        print("❌ 資料庫連線失敗：", err)
        return None


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------- 清洗函式（全域唯一版本）-------
def only_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")


def only_alnum(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s or "")


def only_number(s: str) -> str:
    s = re.sub(r"[^0-9.]", "", s or "")
    if s.count(".") > 1:
        i = s.find(".")
        s = s[:i+1] + s[i+1:].replace(".", "")
    return s


def clean_price(raw):
    if not raw:
        return None
    s = str(raw).replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
    m = re.search(r'(\d[\d,]*\.?\d*)', s)
    if not m:
        return None
    num = m.group(1).replace(',', '')
    try:
        return Decimal(num)
    except InvalidOperation:
        return None


def clean_date_str(raw):
    if not raw:
        return None
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
                "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    m = re.search(r'([A-Za-z]{3,9})\s?(\d{1,2}),?\s?(\d{4})', s)
    if m:
        try:
            return datetime.strptime(' '.join(m.groups()), "%B %d %Y").date().isoformat()
        except ValueError:
            pass
    return None
# ---------------------------------


# ---------- 一般頁面 ----------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/hello')
def hello():
    return "Hello Flask"


# ---------- 進度 API ----------
@app.route('/progress/<job_id>')
def get_progress(job_id):
    if job_id not in PROGRESS:
        return jsonify({"exists": False}), 404
    p = PROGRESS[job_id]
    percent = 0 if p["total"] == 0 else int(p["done"] * 100 / p["total"])
    return jsonify({
        "exists": True,
        "total": p["total"],
        "done": p["done"],
        "percent": percent,
        "finished": p["finished"],
        "error": p["error"]
    })


# ---------- 登入 / 註冊 / 帳戶 ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash("請輸入帳號及密碼")
            return redirect(url_for('login'))

        conn = get_db_connection()
        if not conn:
            flash("資料庫連線失敗")
            return redirect(url_for('login'))

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE us_na = %s", (username,))
            user = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()

        if user and check_password_hash(user['pawd'], password):
            session['user_id'] = user['id']
            session['username'] = user['us_na']
            return redirect(url_for('home'))
        else:
            flash("帳號或密碼錯誤")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        us_na = request.form.get('username', '').strip()
        pawd = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        mail = request.form.get('email', '').strip()
        co_na = request.form.get('company_name', '').strip()
        tax_id = request.form.get('tax_id', '').strip()
        pho = request.form.get('phone', '').strip()
        oer = request.form.get('owner', '').strip()
        ades = request.form.get('address', '').strip()
        desc = request.form.get('description', '').strip()

        if not re.fullmatch(r'[A-Za-z0-9]{8,12}', us_na):
            flash("帳號需為8~12碼英數字")
            return redirect(url_for('register'))
        if not re.fullmatch(r'(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,16}', pawd):
            flash("密碼需為8~16碼英數字，且含英文及數字")
            return redirect(url_for('register'))

        hashed = generate_password_hash(pawd)
        conn = get_db_connection()
        if not conn:
            flash("資料庫連線失敗")
            return redirect(url_for('register'))

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE us_na = %s", (us_na,))
            if cursor.fetchone():
                flash("帳號已存在")
                return redirect(url_for('register'))

            cursor.execute(
                "INSERT INTO users (us_na, pawd, name, mail) VALUES (%s, %s, %s, %s)",
                (us_na, hashed, name, mail)
            )
            user_id = cursor.lastrowid

            if co_na and tax_id:
                cursor.execute("""
                    INSERT INTO companies (user_id, co_na, tax_id, pho, oer, ades, `desc`)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_id, co_na, tax_id, pho, oer, ades, desc))

            conn.commit()
            flash("註冊成功，請登入")
            return redirect(url_for('login'))
        except Exception as e:
            print("註冊錯誤：", e)
            flash("註冊過程發生錯誤")
            return redirect(url_for('register'))
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')


@app.route('/account_edit', methods=['GET', 'POST'])
def account_edit():
    if 'user_id' not in session:
        flash("請先登入")
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash("資料庫連線失敗")
        return redirect(url_for('search'))

    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
        old_pwd = request.form.get('oldPwd', '')
        new_pwd = request.form.get('newPwd', '')
        confirm_pwd = request.form.get('confirmPwd', '')
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        co_na = request.form.get('company_name', '')
        tax_id = request.form.get('tax_id', '')
        pho = request.form.get('phone', '')
        oer = request.form.get('owner', '')
        ades = request.form.get('address', '')
        desc = request.form.get('description', '')

        cursor.execute("SELECT pawd FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user or not check_password_hash(user['pawd'], old_pwd):
            flash("舊密碼錯誤")
            cursor.close()
            conn.close()
            return redirect(url_for('account_edit'))

        if new_pwd:
            if new_pwd != confirm_pwd:
                flash("新密碼與確認密碼不符")
                cursor.close()
                conn.close()
                return redirect(url_for('account_edit'))
            if len(new_pwd) < 8:
                flash("新密碼至少需8碼")
                cursor.close()
                conn.close()
                return redirect(url_for('account_edit'))
            hashed_new_pwd = generate_password_hash(new_pwd)
            cursor.execute("UPDATE users SET pawd = %s WHERE id = %s", (hashed_new_pwd, user_id))

        cursor.execute("UPDATE users SET name=%s, mail=%s WHERE id=%s", (name, email, user_id))

        cursor.execute("SELECT id FROM companies WHERE user_id = %s", (user_id,))
        company = cursor.fetchone()
        if company:
            cursor.execute("""
                UPDATE companies
                SET co_na=%s, tax_id=%s, pho=%s, oer=%s, ades=%s, `desc`=%s
                WHERE user_id=%s
            """, (co_na, tax_id, pho, oer, ades, desc, user_id))
        else:
            cursor.execute("""
                INSERT INTO companies (user_id, co_na, tax_id, pho, oer, ades, `desc`)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, co_na, tax_id, pho, oer, ades, desc))

        conn.commit()
        cursor.close()
        conn.close()
        flash("資料更新成功")
        return redirect(url_for('account_edit'))

    cursor.execute("SELECT us_na, name, mail FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.execute("SELECT * FROM companies WHERE user_id = %s", (user_id,))
    company = cursor.fetchone()
    cursor.close()
    conn.close()

    return render_template('account_edit.html', user=user, company=company)


# ---------- 公司維護 ----------
@app.route('/vendor_manage')
def vendor_manage():
    if 'user_id' not in session:
        flash("請先登入")
        return redirect(url_for("login"))

    conn = get_db_connection()
    if not conn:
        flash("資料庫連線失敗")
        return redirect(url_for("home"))

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, co_na AS name, tax_id, ades AS address, oer AS contact
            FROM companies
            WHERE user_id = %s
        """, (session["user_id"],))
        vendors = cursor.fetchall()
    except Exception as e:
        print("公司資料維護錯誤:", e)
        flash("載入公司資料失敗")
        vendors = []
    finally:
        cursor.close()
        conn.close()

    return render_template("vendor_manage.html", vendors=vendors)


@app.route('/add_vendor', methods=['GET', 'POST'])
def add_vendor():
    if 'user_id' not in session:
        flash("請先登入")
        return redirect(url_for('login'))

    if request.method == 'POST':
        co_na = request.form.get('company_name', '').strip()
        tax_id = request.form.get('tax_id', '').strip()
        pho = request.form.get('phone', '').strip()
        oer = request.form.get('owner', '').strip()
        dtex = request.form.get('dtex', '').strip()

        if not co_na:
            flash("公司名稱為必填")
            return render_template('add_vendor.html')
        if not tax_id:
            flash("統一編號為必填")
            return render_template('add_vendor.html')

        conn = get_db_connection()
        if not conn:
            flash("資料庫連線失敗")
            return render_template('add_vendor.html')

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM companies WHERE tax_id = %s", (tax_id,))
            if cursor.fetchone():
                flash("統一編號已存在")
                return render_template('add_vendor.html')

            cursor.execute("""
                INSERT INTO companies (user_id, co_na, tax_id, pho, oer, dtex)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (session['user_id'], co_na, tax_id, pho, oer, dtex))

            conn.commit()
            flash("新增廠商成功")
            return redirect(url_for('search'))
        except Exception as e:
            print("新增廠商錯誤:", e)
            flash("新增廠商失敗")
            return render_template('add_vendor.html')
        finally:
            cursor.close()
            conn.close()

    return render_template('add_vendor.html')


# ---------- 查詢 ----------
@app.route('/search', methods=['GET', 'POST'])
def search():
    if 'user_id' not in session:
        flash("請先登入")
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        return "資料庫連線失敗"

    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, co_na FROM companies WHERE user_id = %s", (user_id,))
    companies = cursor.fetchall()

    selected_vendor = None
    invoices = []
    date_range = ''

    if request.method == 'POST':
        selected_vendor = request.form.get('vendor')
        date_range = request.form.get('date_range', '').strip()

        query = """
            SELECT in_nu AS num,
                   invoices.tax_id AS snu,
                   in_date AS data,
                   in_pri AS price,
                   co_na AS bun,
                   created_at AS ctime,
                   file_path AS path
            FROM invoices
            JOIN companies ON invoices.tax_id = companies.tax_id
            WHERE invoices.user_id = %s
        """
        params = [user_id]

        if selected_vendor:
            cursor.execute("SELECT tax_id FROM companies WHERE id = %s", (selected_vendor,))
            row = cursor.fetchone()
            if row:
                tax_id = row['tax_id']
                query += " AND invoices.tax_id = %s"
                params.append(tax_id)

        if date_range and ' - ' in date_range:
            try:
                start_date, end_date = [d.strip() for d in date_range.split(' - ')]
                query += " AND in_date BETWEEN %s AND %s"
                params.extend([start_date, end_date])
            except ValueError:
                flash("日期格式錯誤")

        cursor.execute(query, tuple(params))
        invoices = cursor.fetchall()

        for inv in invoices:
            ctime = inv.get('ctime')
            if ctime:
                if isinstance(ctime, datetime):
                    inv['ctime'] = ctime.strftime('%Y-%m-%d %H:%M')
                elif isinstance(ctime, str):
                    try:
                        dt = datetime.fromisoformat(ctime)
                        inv['ctime'] = dt.strftime('%Y-%m-%d %H:%M')
                    except ValueError:
                        inv['ctime'] = ctime

    cursor.close()
    conn.close()

    return render_template('search.html',
                           companies=companies,
                           invoices=invoices,
                           selected_vendor=int(selected_vendor) if selected_vendor else None,
                           date_range=date_range)


# 以公司/期間搜尋（如果你有使用這頁）
@app.route('/invoice_search', methods=['GET'])
def invoice_search():
    company_name = request.args.get('company_name', '')
    tax_id = request.args.get('tax_id', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT c.co_na AS company_name,
               c.tax_id AS company_id,
               i.in_nu  AS invoice_number,
               i.in_pri AS amount,
               i.in_date AS date
        FROM invoices i
        JOIN companies c ON i.tax_id = c.tax_id
        WHERE 1=1
    """
    params = []

    if company_name:
        query += " AND c.co_na LIKE %s"
        params.append(f"%{company_name}%")
    if tax_id:
        query += " AND c.tax_id = %s"
        params.append(tax_id)
    if start_date:
        query += " AND i.in_date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND i.in_date <= %s"
        params.append(end_date)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    return render_template('invoice_search.html',
                           company_name=company_name,
                           tax_id=tax_id,
                           start_date=start_date,
                           end_date=end_date,
                           results=results)


# ---------- 手動新增 ----------
@app.route('/manual_invoice', methods=['GET', 'POST'])
def manual_invoice():
    if 'user_id' not in session:
        flash('請先登入才能登錄發票')
        return redirect(url_for('login'))

    if request.method == 'POST':
        taxid = request.form['taxid']
        invoice_num = request.form['invoice_num']
        date = request.form['date']
        amount = request.form['amount']
        file = request.files['upload']

        file_path = ''
        if file and file.filename:
            filename = secure_filename(file.filename)
            upload_folder = os.path.join(app.static_folder, 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            save_path = os.path.join(upload_folder, filename)
            file.save(save_path)
            file_path = os.path.join('uploads', filename)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO invoices (in_nu, in_date, in_pri, tax_id, user_id, file_path)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                invoice_num[:50],
                date,
                amount,
                taxid,
                session.get("user_id"),
                file_path
            ))
            conn.commit()
            flash('✅ 發票登錄成功')
        except Exception as e:
            print("❌ 發票登錄失敗：", e)
            flash('❌ 發票登錄失敗，請稍後再試')
        finally:
            cursor.close()
            conn.close()

        return redirect(url_for('manual_invoice'))

    return render_template('manual_invoice.html')


# ---------- 自動辨識 UI ----------
@app.route('/auto_inv')
def invoice_auto():
    return render_template('auto_inv.html')


@app.route('/balance_check')
def balance_check():
    return render_template('check_balance.html')


# ---------- 上傳與辨識 ----------
@app.route('/last_results', methods=['GET'])
def last_results():
    return jsonify(session.get('recognized_results', []))


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': '沒有檔案'}), 400

    job_id = request.form.get('job_id') or str(uuid.uuid4())
    files = request.files.getlist('files')

    PROGRESS[job_id] = {
        "total": 0,
        "done": 0,
        "started_at": datetime.utcnow().isoformat(),
        "finished": False,
        "error": ""
    }

    results = []

    try:
        for file in files:
            try:
                if file.filename == '' or not allowed_file(file.filename):
                    PROGRESS[job_id]["done"] += 1
                    continue

                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)

                image_paths = []
                if filename.lower().endswith('.pdf'):
                    try:
                        images = convert_from_path(save_path, poppler_path=POPPLER_PATH)
                        for i, img in enumerate(images, start=1):
                            img_filename = f"{os.path.splitext(filename)[0]}_{i}.png"
                            img_path = os.path.join(UPLOAD_FOLDER, img_filename)
                            img.save(img_path, 'PNG')
                            image_paths.append(img_path)
                    except Exception as e:
                        print("❌ PDF 轉圖失敗:", e)
                        continue
                else:
                    image_paths.append(save_path)

                PROGRESS[job_id]["total"] += len(image_paths)

                for img_path in image_paths:
                    try:
                        image = cv2.imread(img_path)
                        if image is None:
                            PROGRESS[job_id]["done"] += 1
                            continue

                        parsed_data = process_image(image, os.path.basename(img_path))
                        processed_results[os.path.basename(img_path)] = parsed_data

                        results.append({
                            "num":   parsed_data.get("num", ""),
                            "snu":   parsed_data.get("snu", ""),
                            "data":  parsed_data.get("data", ""),
                            "price": parsed_data.get("price", ""),
                            "bnu":   parsed_data.get("bnu", ""),
                            "name":  parsed_data.get("name", ""),
                            "add":   parsed_data.get("add", ""),
                            "imageUrl": f"/uploads/{os.path.basename(img_path)}",
                            "filename": os.path.basename(img_path)
                        })
                    except Exception as e:
                        print(f"❌ 單張處理失敗 {os.path.basename(img_path)}:", e)
                    finally:
                        PROGRESS[job_id]["done"] += 1

            except Exception as e:
                print(f"❌ 檔案處理失敗 {getattr(file, 'filename', '')}:", e)

        session['recognized_results'] = results
        PROGRESS[job_id]["finished"] = True

        if not results:
            return jsonify({'job_id': job_id, 'error': '未辨識到任何結果'}), 200

        return jsonify({"job_id": job_id, "results": results})

    except Exception as e:
        PROGRESS[job_id]["error"] = str(e)
        PROGRESS[job_id]["finished"] = True
        return jsonify({"job_id": job_id, "error": str(e)}), 500


# ---------- 儲存辨識結果 ----------
@app.route('/confirm_result', methods=['POST'])
def confirm_result():
    if 'user_id' not in session:
        return jsonify({'error': '未登入'}), 403

    rows = request.json
    if not isinstance(rows, list):
        return jsonify({'error': '資料格式錯誤'}), 400

    # 清洗工具
    import re
    from decimal import Decimal, InvalidOperation
    def only_digits(s):  # 只留數字
        return re.sub(r'[^0-9]', '', s or '')
    def only_alnum(s):   # 英數
        return re.sub(r'[^A-Za-z0-9]', '', s or '')
    def clean_price(raw):
        if not raw: return None
        s = str(raw).replace('O','0').replace('o','0').replace('I','1').replace('l','1')
        m = re.search(r'(\d[\d,]*\.?\d*)', s)
        if not m: return None
        num = m.group(1).replace(',', '')
        try:
            return Decimal(num)
        except InvalidOperation:
            return None
    def clean_date(raw):
        if not raw: return None
        s = str(raw).strip()
        for fmt in ("%Y-%m-%d","%Y/%m/%d","%Y.%m.%d","%Y%m%d","%d-%m-%Y","%d/%m/%Y","%d.%m.%Y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                pass
        m = re.search(r'([A-Za-z]{3,9})\s?(\d{1,2}),?\s?(\d{4})', s)
        if m:
            try:
                return datetime.strptime(' '.join(m.groups()), "%B %d %Y").date().isoformat()
            except ValueError:
                pass
        return None

    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '資料庫連線失敗'}), 500
        cur = conn.cursor()

        # 先準備一個「一定建立得出來」的公司 upsert
        # 假設 companies.tax_id 上有 UNIQUE/PK（外鍵會要求它被索引）
        ensure_company_sql = """
            INSERT INTO companies (user_id, co_na, tax_id)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE co_na = co_na
        """

        for r in rows:
            in_nu   = only_alnum((r.get('num') or ''))[:14]
            tax_id  = only_digits((r.get('snu') or ''))[:8]
            in_tax  = only_digits((r.get('bnu') or ''))[:8]
            in_date = clean_date(r.get('data') or None)
            in_pri  = clean_price(r.get('price'))

            # 若有統編，先 upsert 公司，避免外鍵錯誤
            if tax_id:
                cur.execute(ensure_company_sql, (session['user_id'], '(未命名)', tax_id))
                # 先提交，確保這筆被外鍵看得到（不同 MySQL 設定/隔離級別下更保險）
                conn.commit()

            # 寫入發票（允許 None -> 欄位需允許 NULL）
            cur.execute("""
                INSERT INTO invoices (in_nu, in_date, in_pri, tax_id, in_tax, user_id, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                in_nu or None,
                in_date,
                in_pri,
                tax_id or None,
                in_tax or None,
                session['user_id'],
                'auto'
            ))

        conn.commit()
        return jsonify({'message': '✅ 發票資料已成功儲存'})

    except Exception as e:
        try: conn.rollback()
        except: pass
        print("❌ 發票儲存錯誤：", e)
        return jsonify({'error': str(e)}), 500
    finally:
        try: cur.close()
        except: pass
        try: conn.close()
        except: pass

# ---------- OCR 主流程 ----------
def process_image(image, filename):
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(img_path, image)
    print(f"✅ 原圖儲存：{os.path.abspath(img_path)}")

    results = model(image)
    predictions = results.pandas().xywh[0]
    print(f"【DEBUG】{filename} 預測框數：{len(predictions)}")

    data = {
        "imageUrl": f"/uploads/{filename}",
        "num": "",   # 英數
        "snu": "",   # 數字
        "data": "",  # 日期原樣（結果頁顯示；寫 DB 時再 clean）
        "price": "", # 數字.小數
        "bnu": "",   # 數字
        "name": "",  # 公司名（顯示用）
        "add": ""    # 地址（顯示用）
    }

    if predictions.empty:
        print(f"⚠️ {filename} 無預測框")
        return data

    for _, row in predictions.iterrows():
        key_raw = str(row['name'])
        # 類別名清洗：去頭尾空白，只留安全字元（英數、底線、減號）
        key = re.sub(r'[^A-Za-z0-9_-]+', '', key_raw.strip()) or 'field'

        cx, cy, w, h = row['xcenter'], row['ycenter'], row['width'], row['height']
        x1 = max(0, int(cx - w / 2)); y1 = max(0, int(cy - h / 2))
        x2 = min(image.shape[1], int(cx + w / 2)); y2 = min(image.shape[0], int(cy + h / 2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️ 裁切圖為空：{key_raw} -> {key}")
            continue

        cropped_filename = f"{filename.rsplit('.', 1)[0]}_{key}.jpg"
        cv2.imwrite(os.path.join(CROPPED_FOLDER, cropped_filename), crop)

        # 只存檔名
        data[key + "Img"] = cropped_filename

        text = pytesseract.image_to_string(
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            lang='eng+chi_tra',
            config='--psm 6'
        ).strip().replace('\n','').replace(' ','')
        data[key] = text


    # 回傳前做輕量清洗（顯示上更友善；DB 寫入仍以 confirm_result 再嚴格處理）
    data["num"]   = only_alnum(data.get("num"))[:14]
    data["snu"]   = only_digits(data.get("snu"))[:8]
    data["bnu"]   = only_digits(data.get("bnu"))[:8]
    data["price"] = only_number(data.get("price"))

    print("✅ 清洗後：", data)
    return data


# ---------- 上傳目錄 ----------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/uploads/cropped/<filename>')
def uploaded_cropped_file(filename):
    return send_from_directory(CROPPED_FOLDER, filename)


# ---------- 單張結果頁 ----------
@app.route('/result/<filename>')
def result_view(filename):
    result = processed_results.get(filename)
    if not result:
        return "找不到結果", 404

    display_plan = [
        ("num","發票號碼"),("snu","統一編號"),("data","開立日期"),
        ("price","價格"),("bnu","賣方編號"),("name","公司名稱"),("add","公司地址"),
    ]

    texts = []
    base_noext = os.path.splitext(filename)[0]

    for key, label in display_plan:
        val = (result.get(key) or "")
        # 優先用 process_image 記錄的檔名
        raw_rel = (result.get(f"{key}Img") or "").strip().replace("\\", "/")
        crop_name = raw_rel.split("/")[-1] if raw_rel else ""

        # 若沒有或不存在，試著在資料夾用「寬鬆條件」找
        if not crop_name or not os.path.exists(os.path.join(CROPPED_FOLDER, crop_name)):
            pattern = os.path.join(CROPPED_FOLDER, f"{base_noext}_*.jpg")
            candidates = [os.path.basename(p) for p in glob(pattern)]
            key_l = key.lower()
            picked = next((c for c in candidates if key_l in c.lower()), None)
            crop_name = picked or f"{base_noext}_{key}.jpg"  # 仍保留最後備案

        texts.append({
            "key": key,
            "label": label,
            "text": val,
            "cropped_image": crop_name.strip(),  # 一律去空白
        })

    page_data = {"imageUrl": result.get("imageUrl", f"/uploads/{filename}"), "filename": filename}
    return render_template('result.html', result=page_data, texts=texts)

# ---------- 其他頁 ----------
@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    if 'file' not in request.files:
        return jsonify({'error': '沒有收到影像'}), 400

    fs = request.files['file']
    filename = secure_filename(str(uuid.uuid4()) + '.jpg')
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    fs.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        return jsonify({'error': '讀取影像失敗'}), 400

    data = process_image(img, filename)
    processed_results[filename] = data

    result = {
        "num":   data.get("num", ""),
        "snu":   data.get("snu", ""),
        "data":  data.get("data", ""),
        "price": data.get("price", ""),
        "bnu":   data.get("bnu", ""),
        "name":  data.get("name", ""),
        "add":   data.get("add", ""),
        "imageUrl": f"/uploads/{filename}",
        "filename": filename
    }
    return jsonify([result])



# 放在 app.py 其他 route 後面
@app.route('/inv_re', methods=['GET', 'POST'])
def inv_re():
    if 'user_id' not in session:
        flash("請先登入")
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    selected_vendor = None
    start_date = None
    end_date = None
    date_range = ''

    if request.method == 'POST':
        selected_vendor = request.form.get('vendor')
        date_range = request.form.get('date_range', '').strip()

        if date_range and ' - ' in date_range:
            try:
                start_date, end_date = [d.strip() for d in date_range.split(' - ')]
            except ValueError:
                flash("日期格式錯誤")

    # 🟦 共用條件處理
    conditions = ["invoices.user_id = %s"]
    params = [user_id]

    if selected_vendor:
        cursor.execute("SELECT tax_id FROM companies WHERE id = %s", (selected_vendor,))
        row = cursor.fetchone()
        if row:
            conditions.append("invoices.tax_id = %s")
            params.append(row['tax_id'])

    if start_date and end_date:
        conditions.append("in_date BETWEEN %s AND %s")
        params.extend([start_date, end_date])

    condition_str = " AND ".join(conditions)

    # 1️⃣ 每月發票總金額
    cursor.execute(f"""
        SELECT DATE_FORMAT(in_date, '%%Y-%%m') AS month,
               SUM(COALESCE(in_pri, 0)) AS total
        FROM invoices
        WHERE {condition_str}
        GROUP BY month ORDER BY month
    """, tuple(params))
    total_amount_by_month = cursor.fetchall()

    # 2️⃣ 各廠商金額占比
    cursor.execute(f"""
        SELECT companies.co_na AS name,
               SUM(COALESCE(in_pri, 0)) AS total
        FROM invoices
        JOIN companies ON invoices.tax_id = companies.tax_id
        WHERE {condition_str}
        GROUP BY name
    """, tuple(params))
    vendor_amount_ratio = cursor.fetchall()

    # 3️⃣ 發票數量趨勢（按日期統計）
    cursor.execute(f"""
        SELECT in_date AS date,
               COUNT(*) AS count
        FROM invoices
        WHERE {condition_str}
        GROUP BY in_date
        ORDER BY in_date
    """, tuple(params))
    invoice_count_trend = cursor.fetchall()

    # 4️⃣ 發票金額排行（最高5名）
    cursor.execute(f"""
        SELECT companies.co_na AS name,
               SUM(COALESCE(in_pri, 0)) AS total
        FROM invoices
        JOIN companies ON invoices.tax_id = companies.tax_id
        WHERE {condition_str}
        GROUP BY name
        ORDER BY total DESC
        LIMIT 5
    """, tuple(params))
    top_vendors = cursor.fetchall()

    # 5️⃣ 發票金額排行（最低5名）
    cursor.execute(f"""
        SELECT companies.co_na AS name,
               SUM(COALESCE(in_pri, 0)) AS total
        FROM invoices
        JOIN companies ON invoices.tax_id = companies.tax_id
        WHERE {condition_str}
        GROUP BY name
        ORDER BY total ASC
        LIMIT 5
    """, tuple(params))
    bottom_vendors = cursor.fetchall()

    # 6️⃣ 自動 vs 手動數量統計
    cursor.execute(f"""
        SELECT source, COUNT(*) AS count
        FROM invoices
        WHERE {condition_str}
        GROUP BY source
    """, tuple(params))
    source_data = cursor.fetchall()

    source_ratio = {
        'manual': 0,
        'auto': 0
    }
    for row in source_data:
        source_ratio[row['source']] = row['count']

    # 所有該使用者的廠商
    cursor.execute("SELECT id, co_na FROM companies WHERE user_id = %s", (user_id,))
    companies = cursor.fetchall()

    # 🟦 Decimal 轉 float（避免 JSON 錯誤）
    def convert_decimal(data):
        for item in data:
            for k, v in item.items():
                if isinstance(v, Decimal):
                    item[k] = float(v)
        return data

    total_amount_by_month = convert_decimal(total_amount_by_month)
    vendor_amount_ratio = convert_decimal(vendor_amount_ratio)
    invoice_count_trend = convert_decimal(invoice_count_trend)
    top_vendors = convert_decimal(top_vendors)
    bottom_vendors = convert_decimal(bottom_vendors)

    cursor.close()
    conn.close()

    return render_template("inv_re.html",
        companies=companies,
        selected_vendor=int(selected_vendor) if selected_vendor else None,
        start_date=start_date,
        end_date=end_date,
        date_range=date_range,
        total_amount_by_month=total_amount_by_month,
        vendor_amount_ratio=vendor_amount_ratio,
        invoice_count_trend=invoice_count_trend,
        top_vendors=top_vendors,
        bottom_vendors=bottom_vendors,
        source_ratio=source_ratio
    )


# ---------- 偏好 ----------
@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if request.method == 'POST':
        # 略：自行儲存偏好
        flash('偏好設定已更新')
        return redirect(url_for('preferences'))
    return render_template('preferences.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
