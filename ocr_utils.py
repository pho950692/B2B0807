# -*- coding: utf-8 -*-
"""
YOLO 已裁切小圖 → OCR(eng) → 依版型(mi/op/pc)錨點規則擷取
"""
from typing import Dict, Optional, Tuple, Union
import re
import os
import pytesseract

try:
    import cv2
except Exception:
    cv2 = None
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None


# ========== 工具 ==========
_MONTH = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}

# ocr_utils.py
def pdf_to_images(pdf_path: str, dpi: int = 400):
    if convert_from_path is None:
        return []
    try:
        poppler = os.environ.get("POPPLER_PATH")  # 由 core_app.py 設好
        kwargs = {"dpi": dpi}
        if poppler:
            kwargs["poppler_path"] = poppler
        return convert_from_path(pdf_path, **kwargs)
    except Exception:
        return []



def fix_mi_invoice_num(text: str) -> str:
    if not text:
        return text

    t = re.sub(r'[^A-Za-z0-9]', '', text)
    if len(t) < 10:
        return t

    head, tail = t[0], t[1:10]
    if head.isdigit():
        mapping = {"0": "O", "6": "G", "8": "B", "1": "I", "2": "Z"}
        head = mapping.get(head, head)

    head = head.upper()
    tail = re.sub(r'\D', '', tail)[:9]

    return head + tail

# --- 取代原本的 _preprocess 與 _read_as_text ---
def _preprocess(img):
    if img is None: 
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 小圖放大一點（對細字、點陣 PDF 很有幫助）
    h, w = g.shape[:2]
    scale = 3 if max(h, w) < 300 else 2
    g = cv2.resize(g, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # 提升對比（CLAHE）+ 二值化 + 去噪
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    g = cv2.medianBlur(g, 3)
    return g

def _read_as_text(img_path: str, lang: str = "eng", config: str = "") -> str:
    if pytesseract is None or cv2 is None:
        return ""
    img = cv2.imread(img_path)
    if img is None:
        return ""
    proc = _preprocess(img)
    if proc is None:
        return ""
    # 預設採用 --oem 1、--psm 6，適合單欄小區塊
    base_cfg = "--oem 1 --psm 6"
    if config:
        base_cfg = f"{base_cfg} {config}"
    t = pytesseract.image_to_string(proc, lang=lang, config=base_cfg)
    return (t or "").strip()

# 針對數字/英數欄位的便捷讀取
def _read_digits(img_path: str) -> str:
    return _read_as_text(img_path, lang="eng", config="tessedit_char_whitelist=0123456789")

def _read_alnum(img_path: str) -> str:
    return _read_as_text(
        img_path, lang="eng",
        config="tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
    )


def _after_anchor(
    text: str,
    anchor_pat: str,
    value_pat: Optional[str],
    flags=re.I,
    window: int = 300
) -> str:
    """在錨點後的固定視窗內取值。"""
    if not text:
        return ""
    m = re.search(anchor_pat, text, flags)
    if not m:
        return ""
    seg = text[m.end(): m.end()+window]
    if value_pat:
        mv = re.search(value_pat, seg, flags)
        return mv.group(0).strip() if mv else ""
    return seg.strip()


def _parse_date(s: str) -> str:
    if not s: return ""
    s = s.strip()
    # yyyy-mm-dd / yyyy.mm.dd / yyyy/mm/dd
    m = re.search(r"(\d{4})[.\-\/](\d{1,2})[.\-\/](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{y:04d}/{mo:02d}/{d:02d}"
    # Mar 14, 2020
    m = re.search(r"([A-Za-z]{3,})\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        mo = _MONTH.get(m.group(1).lower()[:3], 0)
        if mo:
            return f"{int(m.group(3)):04d}/{mo:02d}/{int(m.group(2)):02d}"
    # dd/mm/yyyy
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        return f"{y:04d}/{mo:02d}/{d:02d}"
    return ""

def _clean_year_range(yyyy_mm_dd: str) -> str:
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})$", yyyy_mm_dd or "")
    if not m: return ""
    y = int(m.group(1))
    return yyyy_mm_dd if 1900 <= y <= 2100 else ""

def _clean_price(s: str) -> str:
    if not s: return ""
    s = (s.splitlines() or [""])[0]
    s = re.sub(r"[A-Za-z$€£¥NTWDusd,\s]", "", s)
    m = re.search(r"\d+(?:\.\d+)?", s)
    return m.group(0) if m else ""

def _keep_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def _deconfuse_alnum_for_op(s: str) -> str:
    """OP 編號常見誤讀修正：字母夾著的 0/9 視為 O。"""
    if not s: return s
    # 例如 L9X -> LOX、AB0C -> ABOC
    return re.sub(r"(?<=[A-Z])[09](?=[A-Z])", "O", s)

def _clean_num(raw: str, inv_type: str) -> str:
    raw = (raw or "").strip().upper()
    if not raw:
        return ""
    if inv_type == "mi":
        s = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())

        if not s:
            return ""

        # ✅ 如果第一碼是數字 → 依照規則轉換
        if s[0].isdigit():
            mapping = {"0": "O", "6": "G", "8": "B", "1": "I", "2": "Z"}
            first = mapping.get(s[0], s[0])
        else:
            first = s[0]

        # 後 9 碼只保留數字
        digits = re.sub(r"\D", "", s[1:])
        digits = digits[:9]

        if len(digits) < 9:
            return first + digits  # 不足9碼就返回現有

        return first + digits

    if inv_type == "op":
        # 專門抓 "英數混合-英數混合" 格式 (至少3碼-至少2碼)
        m = re.search(r"[A-Z0-9]{3,}-[A-Z0-9]{2,}", raw)
        if m:
            fixed = _deconfuse_alnum_for_op(m.group(0))
            return fixed

        # 備援：允許至少 6 碼以上的英數混合 (避免沒抓到)
        m = re.search(r"[A-Z0-9\-]{6,}", raw)
        if m:
            fixed = _deconfuse_alnum_for_op(m.group(0))
            return fixed
    if inv_type == "pc":
        # 放寬 PC 發票號碼格式：英數字 + 可含破折號，至少 6 碼
        m = re.search(r"[A-Z0-9\-]{6,}", raw)
        if m:
            return m.group(0)


def _clean_sun(raw: str) -> str:
    m = re.search(r"(?<!\d)\d{8}(?!\d)", raw or "")
    return m.group(0) if m else _keep_digits(raw)[:8]

def _clean_date(raw: str) -> str:
    return _clean_year_range(_parse_date(raw or ""))

def _clean_cash(raw: str, inv_type: str = "") -> str:
    if not raw:
        return ""
    if inv_type == "pc":
        # PC：抓「元」之前的數字，容忍 "3, 600元" / "3,600元"
        m = re.search(r"([\d,\s]+)\s*元", raw)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            return digits.lstrip("0") or "0"
    # 其他共用
    s = re.sub(r"[^\d.,]", "", raw)
    m = re.search(r"\d{1,3}(?:[,\d]{0,})", s)
    if not m:
        return ""
    return m.group(0).replace(",", "")


# ========== 版型錨點規則 ==========
# 規則 tuple 支援 (anchor, value) 或 (anchor, value, window)
_MI_RULES: Dict[str, Union[Tuple[str,str], Tuple[str,str,int]]] = {
    "num":  (r"\binvoice\s*number\b",              r"[A-Z0-9\-]+"),
    "date": (r"\binvoice\s*date\s*in\s*utc\b",     r"[A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}"),
    "sun":  (r"\bvat\s*reg\.\s*no\.?\b",           r"\d{6,12}"),
    "cash": (r"\btotal\s*amount\s*twd\b",          r"\d[\d,]*(?:\.\d+)?"),
}

_OP_RULES: Dict[str, Union[Tuple[str,str], Tuple[str,str,int]]] = {
    "num":  (r"\b(invoicenumber|invoice\s*number)\b", r"[A-Z0-9\-]+"),
    "date": (r"\bdate\s*due\b",                        r"[A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}"),
    "sun":  (r"\bvat\b",                               r"\d{6,12}"),
    "cash": (r"\bamount\s*due\b",                      r"\d[\d,]*(?:\.\d+)?"),
}

_PC_RULES: Dict[str, Union[Tuple[str,str], Tuple[str,str,int]]] = {
    # 發票號碼：允許「發票號碼」或「發 票 號 碼」
    "num":  (r"(發\s*票\s*號\s*碼|發票號碼)[:：]?\s*", r"[A-Z0-9]{2}\d{8}|[A-Z0-9\-]+"),

    # 開立日期：允許「開立日期」或「開 立 日 期」
    "date": (r"(開\s*立\s*日\s*期|開立日期)[:：]?\s*", r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}"),

    # 統一編號：允許「統一編號」或「統 一 編 號」
    "sun":  (r"(統\s*一\s*編\s*號|統一編號)[:：]?\s*", r"\d{8}"),

    # 交易金額：允許「交易金額」或「交 易 金 額」
    "cash": (r"(交\s*易\s*金\s*額|交易金額)[:：]?\s*", r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", 120),
}


def _apply_rules(text: str, inv_type: str) -> Dict[str, str]:
    inv = (inv_type or "").lower()
    rules = _MI_RULES if inv == "mi" else _OP_RULES if inv == "op" else _PC_RULES
    out = {}
    for key, rule in rules.items():
        if len(rule) == 2:
            anchor, vr = rule  # type: ignore
            raw = _after_anchor(text, anchor, vr)
        else:
            anchor, vr, win = rule  # type: ignore
            raw = _after_anchor(text, anchor, vr, window=win)
        out[key] = raw if raw else ""
    out["num"]  = _clean_num(out.get("num",""), inv)
    out["sun"]  = _clean_sun(out.get("sun",""))
    out["cash"] = _clean_cash(out.get("cash",""), inv)
    out["date"] = _clean_date(out.get("date",""))
    return out


def ocr_fields_from_crops(crops: Dict[str, str], inv_type: str) -> Dict[str, str]:
    inv = (inv_type or "pc").lower()
    out = {"num":"", "date":"", "sun":"", "cash":""}

    # 先把四塊拼成一個大文本，配你已經寫好的錨點規則跑一次
    segs = []
    for k in ("num","date","sun","cash"):
        p = crops.get(k)
        if p:
            # 大文本用一般英文字元，不限白名單（保留 anchor）
            lang_pool = "eng" if inv in ("mi", "op") else "chi_tra+eng"
            segs.append(_read_as_text(p, lang=lang_pool))
    pool = "\n".join([s for s in segs if s])
    if pool:
        out.update(_apply_rules(pool, inv))

    # 接著逐欄位補強：針對各欄位使用對應的 tesseract 白名單
    if crops.get("num") and not out["num"]:
        if inv in ("mi", "op"):
            out["num"] = _clean_num(_read_alnum(crops["num"]), inv)
        else:  # pc
            out["num"] = _clean_num(_read_alnum(crops["num"]), inv)

    if crops.get("sun") and not out["sun"]:
        out["sun"] = _clean_sun(_read_digits(crops["sun"]))

    if crops.get("cash") and not out["cash"]:
        # 金額允許逗點與小數點，先用英數白名單抓，再交給 _clean_cash
        cash_raw = _read_as_text(crops["cash"], lang="eng",
                                 config="tessedit_char_whitelist=0123456789.,元NTWD")
        out["cash"] = _clean_cash(cash_raw, inv)

    if crops.get("date") and not out["date"]:
        # 日期保留英文字母（月名）與分隔符號
        date_raw = _read_as_text(crops["date"], lang="eng",
                                 config="tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/.-, ")
        out["date"] = _clean_date(date_raw)

    return out

def fullpage_anchor_ocr(img_bgr, inv_type: str):
    if cv2 is None:
        return {"num":"", "date":"", "sun":"", "cash":""}
    inv = (inv_type or "").lower()
    # PDF/英文化的票種用英文；台灣 PC 若要備援可用 chi_tra+eng
    lang = "eng" if inv in ("mi", "op") else "chi_tra+eng"
    proc = _preprocess(img_bgr)
    txt = pytesseract.image_to_string(proc, lang=lang, config="--oem 1 --psm 6") if proc is not None else ""
    out = _apply_rules(txt or "", inv)
    # 再跑一次清洗，確保格式
    out["num"]  = _clean_num(out.get("num",""), inv)
    out["sun"]  = _clean_sun(out.get("sun",""))
    out["cash"] = _clean_cash(out.get("cash",""), inv)
    out["date"] = _clean_date(out.get("date",""))
    return out
