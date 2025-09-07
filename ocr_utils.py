# -*- coding: utf-8 -*-
"""
YOLO 已裁切小圖 → OCR(eng) → 依版型(mi/op/pc)錨點規則擷取
"""
from typing import Dict, Optional, Tuple, Union
import re
import os
import pytesseract

# --- Tesseract 路徑（依你機器） ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

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

def pdf_to_images(pdf_path: str, dpi: int = 300):
    if convert_from_path is None: return []
    try: return convert_from_path(pdf_path, dpi=dpi)
    except Exception: return []

def _preprocess(img):
    if img is None: return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (g.shape[1]*2, g.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    g = cv2.medianBlur(g, 3)
    return g

def _read_as_text(img_path: str, lang: str = "eng") -> str:
    if pytesseract is None or cv2 is None: return ""
    img = cv2.imread(img_path)
    if img is None: return ""
    proc = _preprocess(img)
    if proc is None: return ""
    t = pytesseract.image_to_string(proc, lang=lang)
    return (t or "").strip()


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
        # 若 OCR 錯把 O 輸出成 T，就轉回 O
        if raw.startswith("T") and raw[1:].isdigit():
            return "O" + raw[1:]
        m = re.search(r"[A-Z]\d{5,}", raw)
        if m:
            return m.group(0)
        if raw.isdigit():
            return "O" + raw  # fallback 補 O
    if inv_type == "op":
        m = re.search(r"[A-Z0-9\-]{6,}", raw)
        if m:
            fixed = _deconfuse_alnum_for_op(m.group(0))
            return fixed
    if inv_type == "pc":
        m = re.search(r"[A-Z]{2}\d{8}", raw)
        if m:
            return m.group(0)
    return raw


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
    "num":  (r"發\s*票\s*號\s*碼\s*[:：]\s*", r"[A-Z0-9]{2}\d{8}|[A-Z0-9\-]+"),
    "date": (r"開\s*立\s*日\s*期\s*[:：]\s*", r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}"),
    "sun":  (r"統\s*一\s*編\s*號\s*[:：]\s*", r"\d{8}"),
    # 只取「元」之前，允許逗點與空白，並縮小 window 避免吃到後面欄位
    "cash": (r"交\s*易\s*金\s*額\s*[:：]\s*", r"([\d,\s]+)\s*元", 50),
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

    segs = []
    for k in ("num","date","sun","cash"):
        p = crops.get(k)
        if p: segs.append(_read_as_text(p, lang="eng"))
    pool = "\n".join([s for s in segs if s])

    if pool:
        out.update(_apply_rules(pool, inv))

    if not out["num"]  and crops.get("num"):
        out["num"]  = _clean_num(_read_as_text(crops["num"],  lang="eng"), inv)
    if not out["date"] and crops.get("date"):
        out["date"] = _clean_date(_read_as_text(crops["date"], lang="eng"))
    if not out["sun"]  and crops.get("sun"):
        out["sun"]  = _clean_sun(_read_as_text(crops["sun"],  lang="eng"))
    if not out["cash"] and crops.get("cash"):
        out["cash"] = _clean_cash(_read_as_text(crops["cash"], lang="eng"), inv)

    return out
