import imaplib
import email
from email.header import decode_header
import os
import re
from datetime import datetime

# ====== 使用前請先設定下列資訊 ======
IMAP_SERVER = 'imap.gmail.com'  # Gmail IMAP 伺服器
EMAIL_ACCOUNT = 'pho950692@gmail.com'  # 收件信箱
EMAIL_PASSWORD = 'cgmk ghmb cdsa xtkr'   # 建議用App Password
UPLOAD_DIR = 'uploads'            # 儲存附件的資料夾
# ===================================

# 你要監控的寄件人（只抓指定 email）
COMPANY_SENDERS = {
    '合作公司': re.compile(r's11114147@gm.cyut.edu.tw', re.I),
}

# 允許的附件副檔名
ALLOWED_EXTS = {'.pdf', '.jpg', '.jpeg', '.png'}

def connect_mail():
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
    return mail

def decode_str(s):
    if not s:
        return ''
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='ignore')
    dh = decode_header(s)
    return ''.join([
        (t.decode(enc or 'utf-8') if isinstance(t, bytes) else t)
        for t, enc in dh
    ])

def save_attachment(part, save_path):
    with open(save_path, 'wb') as f:
        f.write(part.get_payload(decode=True))

def fetch_invoices():
    mail = connect_mail()
    mail.select('INBOX')
    # 只抓近 30 天內未讀郵件
    typ, data = mail.search(None, '(UNSEEN SINCE "01-Sep-2025")')
    for num in data[0].split():
        typ, msg_data = mail.fetch(num, '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                sender = decode_str(msg.get('From', ''))
                subject = decode_str(msg.get('Subject', ''))
                date_str = msg.get('Date', '')
                try:
                    date_obj = email.utils.parsedate_to_datetime(date_str)
                except Exception:
                    date_obj = datetime.now()
                date_fmt = date_obj.strftime('%Y%m%d')
                # 判斷是哪家公司
                company = None
                for cname, pat in COMPANY_SENDERS.items():
                    if pat.search(sender):
                        company = cname
                        break
                if not company:
                    continue  # 非目標公司
                # 處理附件
                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition') is None:
                        continue
                    filename = decode_str(part.get_filename())
                    if not filename:
                        continue
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in ALLOWED_EXTS:
                        continue
                    safe_name = f"{company}_{date_fmt}_{filename}"
                    save_path = os.path.join(UPLOAD_DIR, safe_name)
                    save_attachment(part, save_path)
                    print(f"已下載: {save_path}")
    mail.logout()

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    fetch_invoices()
    print('郵件附件抓取完成')
