from flask import Flask, request, render_template, redirect, url_for, flash, session,jsonify, render_template, send_from_directory
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from config import db_config
import re
import os
from werkzeug.utils import secure_filename
import cv2
import torch
import pytesseract
from PIL import Image
from collections import namedtuple
from pdf2image import convert_from_path
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 建議改成環境變數或更強隨機字串

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PDF 轉圖工具
POPPLER_PATH = os.path.join(BASE_DIR, 'poppler', 'Library', 'bin')

# 圖片上傳 & 裁切資料夾
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CROPPED_FOLDER = os.path.join(UPLOAD_FOLDER, 'cropped')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

# OCR 執行檔位置
pytesseract.pytesseract.tesseract_cmd =  r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#

weight_path = r'C:\b2b0806\b2b_st\weights\best.pt'  # 或用 os.path.join(BASE_DIR, 'weights', 'best.pt')

import os
if not os.path.exists(weight_path):
    raise FileNotFoundError(f"❌ 找不到模型權重檔案：{weight_path}")

model = torch.hub.load('ultralytics/yolov5:v6.2', 'custom',
                       path=weight_path,
                       force_reload=True)


def get_db_connection():
    print("嘗試連線資料庫")
    try:
        conn = mysql.connector.connect(**db_config)
        print("✅ 成功連線資料庫")
        return conn
    except mysql.connector.Error as err:
        print("❌ 資料庫連線失敗，錯誤內容如下：")
        print(err)  # 這行會印出詳細錯誤
        return None

@app.route('/')
def home():
    return render_template('home.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/search', methods=['GET', 'POST'])
def search():
    conn = get_db_connection()
    if not conn:
        return "資料庫連線失敗"

    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM companies")
    companies = cursor.fetchall()

    selected_vendor = None
    invoices = []
    date_range = ''

    if request.method == 'POST':
        selected_vendor = request.form.get('vendor')
        date_range = request.form.get('date_range', '').strip()

        query = """
            SELECT in_date AS date, in_nu AS number, in_pri AS amount
            FROM invoices
            WHERE 1=1
        """
        params = []

        if selected_vendor:
            cursor.execute("SELECT tax_id FROM companies WHERE id = %s", (selected_vendor,))
            row = cursor.fetchone()
            if row:
                tax_id = row['tax_id']
                query += " AND tax_id = %s"
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

    cursor.close()
    conn.close()

    return render_template('search.html',
                           companies=companies,
                           invoices=invoices,
                           selected_vendor=int(selected_vendor) if selected_vendor else None,
                           date_range=date_range)


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
        except Exception as e:
            print(f"登入錯誤 Exception: {e}")
            flash("登入過程發生錯誤")
            return redirect(url_for('login'))
        finally:
            cursor.close()
            conn.close()

        if user and check_password_hash(user['pawd'], password):
            session['user_id'] = user['id']
            session['username'] = user['us_na']
            return redirect(url_for('home'))  # ✅ 導向首頁
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

        # 帳號與密碼格式檢查
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
            # 檢查帳號是否存在
            cursor.execute("SELECT 1 FROM users WHERE us_na = %s", (us_na,))
            if cursor.fetchone():
                flash("帳號已存在")
                return redirect(url_for('register'))

            # 新增使用者
            cursor.execute("INSERT INTO users (us_na, pawd, name, mail) VALUES (%s, %s, %s, %s)",
                           (us_na, hashed, name, mail))
            user_id = cursor.lastrowid

            # 若有填廠商資料則新增
            if co_na and tax_id:
                cursor.execute("""
                    INSERT INTO companies (user_id, co_na, tax_id, pho, oer, ades, `desc`)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_id, co_na, tax_id, pho, oer, ades, desc))

            conn.commit()
            flash("註冊成功，請登入")
            return redirect(url_for('login'))

        except Exception as e:
            print(f"註冊錯誤: {e}")
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

        # 先檢查舊密碼是否正確
        cursor.execute("SELECT pawd FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user or not check_password_hash(user['pawd'], old_pwd):
            flash("舊密碼錯誤")
            cursor.close()
            conn.close()
            return redirect(url_for('account_edit'))

        # 新密碼驗證
        if new_pwd:
            if new_pwd != confirm_pwd:
                flash("新密碼與確認密碼不符")
                cursor.close()
                conn.close()
                return redirect(url_for('account_edit'))
            # 可加入強度檢查，這裡簡單示範
            if len(new_pwd) < 8:
                flash("新密碼至少需8碼")
                cursor.close()
                conn.close()
                return redirect(url_for('account_edit'))
            hashed_new_pwd = generate_password_hash(new_pwd)
            cursor.execute("UPDATE users SET pawd = %s WHERE id = %s", (hashed_new_pwd, user_id))

        # 更新 users 其他欄位
        cursor.execute("UPDATE users SET name=%s, mail=%s WHERE id=%s", (name, email, user_id))

        # 更新 companies 資料（假設一個帳號只有一間公司）
        cursor.execute("SELECT id FROM companies WHERE user_id = %s", (user_id,))
        company = cursor.fetchone()
        if company:
            cursor.execute("""
                UPDATE companies SET co_na=%s, tax_id=%s, pho=%s, oer=%s, ades=%s, `desc`=%s WHERE user_id=%s
            """, (co_na, tax_id, pho, oer, ades, desc, user_id))
        else:
            # 如果沒有廠商資料就新增一筆
            cursor.execute("""
                INSERT INTO companies (user_id, co_na, tax_id, pho, oer, ades, `desc`)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, co_na, tax_id, pho, oer, ades, desc))

        conn.commit()
        cursor.close()
        conn.close()
        flash("資料更新成功")
        return redirect(url_for('account_edit'))

    # GET 時帶入資料
    cursor.execute("SELECT us_na, name, mail FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.execute("SELECT * FROM companies WHERE user_id = %s", (user_id,))
    company = cursor.fetchone()
    cursor.close()
    conn.close()

    return render_template('account_edit.html',
                           user=user,
                           company=company)

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
        category = request.form.get('category', '').strip()

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
            # 檢查統一編號是否已存在
            cursor.execute("SELECT 1 FROM companies WHERE tax_id = %s", (tax_id,))
            if cursor.fetchone():
                flash("統一編號已存在")
                return render_template('add_vendor.html')

            # 新增廠商
            cursor.execute("""
                INSERT INTO companies (user_id, co_na, tax_id, pho, oer, category)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (session['user_id'], co_na, tax_id, pho, oer, category))

            conn.commit()
            flash("新增廠商成功")
            return redirect(url_for('search'))

        except Exception as e:
            print(f"新增廠商錯誤: {e}")
            flash("新增廠商失敗")
            return render_template('add_vendor.html')
        finally:
            cursor.close()
            conn.close()

    return render_template('add_vendor.html')



@app.route('/invoice_search', methods=['GET'])
def invoice_search():
    company_name = request.args.get('company_name', '')
    tax_id = request.args.get('tax_id', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT c.co_na AS company_name, c.tax_id AS company_id, i.number AS invoice_number, i.amount, i.date FROM invoices i JOIN companies c ON i.tax_id = c.tax_id WHERE 1=1"
    params = []

    if company_name:
        query += " AND c.co_na LIKE %s"
        params.append(f"%{company_name}%")
    if tax_id:
        query += " AND c.tax_id = %s"
        params.append(tax_id)
    if start_date:
        query += " AND i.date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND i.date <= %s"
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

@app.route('/manual_invoice', methods=['GET', 'POST'])
def manual_invoice():
    if 'user_id' not in session:
        flash('請先登入才能登錄發票')
        return redirect(url_for('login'))

    if request.method == 'POST':
        company = request.form['company']
        taxid = request.form['taxid']
        invoice_num = request.form['invoice_num']
        date = request.form['date']
        amount = request.form['amount']
        file = request.files['upload']

        # 儲存上傳檔案
        file_path = ''
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

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
            print(f"❌ 發票登錄失敗：{e}")
            flash('❌ 發票登錄失敗，請稍後再試')
        finally:
            cursor.close()
            conn.close()

        return redirect(url_for('manual_invoice'))

    return render_template('manual_invoice.html')


@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        tax_id = request.form.get('tax_id')
        phone = request.form.get('phone')
        username = request.form.get('username')
        email = request.form.get('email')
        retention_days = request.form.get('retention_days')
        auto_title = 'auto_title' in request.form
        language = request.form.get('language')
        dark_mode = 'dark_mode' in request.form
        auto_email = 'auto_email' in request.form
        notify_complete = 'notify_complete' in request.form
        notify_error = 'notify_error' in request.form

        # 更新使用者偏好邏輯，例如寫入資料庫...

        flash('偏好設定已更新')
        return redirect(url_for('preferences'))

    return render_template('preferences.html')

@app.route('/auto_inv')
def invoice_auto():
    return render_template('auto_inv.html')

@app.route('/balance_check')
def balance_check():
    # 你的程式邏輯
    return render_template('check_balance.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/hello')
def hello():
    return "Hello Flask"

# ✅ 用來儲存處理結果
processed_results = {}

FIELD_LABELS = {
    0: "發票號碼",
    1: "統一編號",
    2: "日期",
    3: "價格",
    4: "賣方編號",
    5: "公司名稱",
    6: "公司地址"
}


def parse_invoice(filepath):
    # 這裡用你自己模型的辨識邏輯，以下是範例
    from PIL import Image
    import pytesseract

    img = Image.open(filepath)
    text = pytesseract.image_to_string(img, lang='eng+chi_tra')

    # 假設你用正則表達式或其他方式提取欄位
    return {
        '發票號碼': 'AB12345678',
        '統一編號': '12345678',
        '日期': '2025-07-28',
        '價格': '3000',
        '賣方編號': '87654321'
    }

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': '沒有檔案'}), 400

    files = request.files.getlist('files')
    results = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        image_paths = []

        if filename.lower().endswith('.pdf'):
            try:
                images = convert_from_path(save_path, poppler_path=POPPLER_PATH)
                for i, img in enumerate(images):
                    img_filename = f"{filename}_{i}.png"
                    img_path = os.path.join(UPLOAD_FOLDER, img_filename)
                    img.save(img_path, 'PNG')
                    image_paths.append(img_path)
            except Exception as e:
                print(f"❌ PDF 轉圖失敗: {e}")
                continue
        else:
            image_paths.append(save_path)

        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            parsed_data = process_image(image, os.path.basename(img_path))
            processed_results[os.path.basename(img_path)] = parsed_data

            result = {
                "num": parsed_data.get("num", ""),
                "snu": parsed_data.get("snu", ""),
                "data": parsed_data.get("data", ""),
                "price": parsed_data.get("price", ""),
                "bnu": parsed_data.get("bnu", ""),
                "name": parsed_data.get("name", ""),
                "add": parsed_data.get("add", ""),
                "imageUrl": f"/uploads/{os.path.basename(img_path)}"
            }
            results.append(result)

    if not results:
        return jsonify({'error': '未辨識到任何結果'}), 200

    return jsonify(results)


@app.route('/confirm_result', methods=['POST'])
def confirm_result():
    if 'user_id' not in session:
        return jsonify({'error': '未登入'}), 403

    data = request.json
    if not isinstance(data, list):
        return jsonify({'error': '資料格式錯誤'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for row in data:
            cursor.execute("""
                INSERT INTO invoices (in_nu, in_date, in_pri, tax_id, in_tax, user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                row.get("num", "").strip()[:14],  # ⚠️ 限制發票號碼長度
                row.get("data", "").strip(),
                row.get("price", "").strip(),
                row.get("snu", "").strip(),
                row.get("bnu", "").strip(),
                session.get("user_id")
            ))

        conn.commit()
        return jsonify({'message': '✅ 發票資料已成功儲存'})
    except Exception as e:
        print("❌ 發票儲存錯誤：", e)
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

def process_image(image, filename):
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(img_path, image)
    print(f"✅ 原圖儲存路徑：{os.path.abspath(img_path)}")

    results = model(image)
    predictions = results.pandas().xywh[0]

    print(f"【DEBUG】圖片 {filename} 預測框數量：{len(predictions)}")

    data = {
        "imageUrl": f"/uploads/{filename}",
        "num": "",
        "snu": "",
        "data": "",  # 發票日期
        "price": "",
        "bnu": "",
        "name": "",
        "add": ""
    }

    if predictions.empty:
        print(f"⚠️ 圖片 {filename} 無預測框")
        return data

    for idx, row in predictions.iterrows():
        key = row['name']

        cx, cy, w, h = row['xcenter'], row['ycenter'], row['width'], row['height']
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(image.shape[1], int(cx + w / 2))
        y2 = min(image.shape[0], int(cy + h / 2))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️ 裁切圖為空：{key}")
            continue

        cropped_filename = f"{filename.rsplit('.', 1)[0]}_{key}.jpg"
        cropped_path = os.path.join(CROPPED_FOLDER, cropped_filename)
        cv2.imwrite(cropped_path, crop)
        print(f"✅ 裁切圖儲存路徑: {cropped_path}")

        text = pytesseract.image_to_string(
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            lang='eng+chi_tra',
            config='--psm 6'
        ).strip().replace('\n', '').replace(' ', '')

        print(f"🔠 OCR【{key}】：{text}")
        data[key] = text
        data[key + "Img"] = f"uploads/cropped/{cropped_filename}"

    print("✅ 完整結果：", data)
    return data

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    if 'file' not in request.files:
        return jsonify({'error': '沒有收到影像'}), 400

    file = request.files['file']
    filename = secure_filename(str(uuid.uuid4()) + '.jpg')
    save_dir = os.path.join('uploads')
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    file.save(file_path)

    # 呼叫你的 YOLO + OCR 辨識函式
    result = [process_image(cv2.imread(file_path), filename)]  # 假設 process_image 是你之前定義的

    return jsonify(result)

@app.route("/vendor_manage")
def vendor_manage():
    return render_template("vendor_manage.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/uploads/cropped/<filename>')
def uploaded_cropped_file(filename):
    return send_from_directory(CROPPED_FOLDER, filename)

@app.route('/result/<filename>')
def result_view(filename):
    result = processed_results.get(filename)
    if not result:
        return "找不到結果", 404

    Result = namedtuple('Result', result.keys())
    result_obj = Result(*result.values())

    # 直接使用模型原始欄位，不使用 KEY_MAP
    key_label_map = {
        'num': '發票號碼',
        'snu': '統一編號',
        'data': '開立日期',
        'price': '價格',
        'bnu': '賣方編號',
        'name': '公司名稱',
        'add': '公司地址'
    }

    texts = []
    for key, label in key_label_map.items():
        text = getattr(result_obj, key, '')
        cropped_image = getattr(result_obj, key + 'Img', '')
        texts.append({
            'label': label,
            'text': text,
            'cropped_image': cropped_image
        })

    return render_template('result.html', result=result_obj, texts=texts)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
