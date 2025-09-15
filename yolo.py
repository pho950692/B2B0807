# -*- coding: utf-8 -*-
"""
YOLOv5 偵測 + 裁切
流程：
  1) 用 tr3.pt 判斷發票類型：pc / op / mi
  2) 用對應的 pc.pt / op.pt / mi.pt 偵測欄位框：num/date/sun/cash
  3) 依框裁切小圖，交由 ocr_utils 做 OCR + 規則清洗
  4) 回傳欄位文字與裁切小圖路徑（含 web_path，前端可直接 <img src=...>）

需要：torch, opencv-python, pytesseract, pdf2image
"""
from typing import Any, Dict, Optional, List, Tuple
import os
import uuid
from yocr.ocr_utils import ocr_fields_from_crops, fullpage_anchor_ocr
# 依賴
try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None


# ---------- 模型路徑 ----------
def _env_or_default(name: str, filename: str) -> str:
    """
    先讀環境變數，如果沒有就用專案根目錄下的 weights 資料夾。
    """
    p = os.environ.get(name, "")
    if p and os.path.isfile(p):
        return p

    # fallback：專案根目錄下的 weights
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)  # b2b_st 專案根
    weights_dir = os.path.join(root, "weights")
    alt = os.path.join(weights_dir, filename)
    return alt if os.path.isfile(alt) else filename


MODEL_PATHS = {
    "tr3": _env_or_default("YOLO_TR3", "tr3.pt"),
    "pc":  _env_or_default("YOLO_PC",  "pc.pt"),
    "op":  _env_or_default("YOLO_OP",  "op.pt"),
    "mi":  _env_or_default("YOLO_MI",  "mi.pt"),
}

DEFAULT_KEY_ORDER = ["num", "date", "sun", "cash"]

# 啟動時印出路徑確認
print("[YOLO MODEL PATHS]", MODEL_PATHS)


# ---------- 小工具 ----------
def _load_bgr(img_or_path: Any):
    if cv2 is None:
        return None
    if isinstance(img_or_path, (str, bytes)):
        return cv2.imread(img_or_path)
    if getattr(img_or_path, "shape", None) is not None:
        return img_or_path
    return None


def _ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _load_yolo_model(model_path: str):
    if torch is None:
        raise RuntimeError("PyTorch 未安裝，無法載入 YOLOv5 模型。")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到 YOLO 模型：{model_path}")
    # 先試 local，沒有再用 github（需要 Git/可連外）
    try:
        return torch.hub.load("ultralytics/yolov5", "custom",
                              path=model_path, source="local", force_reload=False)
    except Exception:
        return torch.hub.load("ultralytics/yolov5", "custom",
                              path=model_path, source="github", force_reload=False)


def _map_class_to_key(model) -> Dict[int, str]:
    names = getattr(model, "names", None)
    mapping: Dict[int, str] = {}

    def norm_to_key(n_lower: str) -> str:
        if any(k in n_lower for k in ("num", "number", "invno", "invoice number", "invoice_number", "發票號碼")):
            return "num"
        if any(k in n_lower for k in ("date", "日期", "開立日期")):
            return "date"
        if any(k in n_lower for k in ("sun", "vat", "統一編號")):
            return "sun"
        if any(k in n_lower for k in ("cash", "amount", "price", "total", "交易金額", "金額")):
            return "cash"
        if n_lower in ("pc", "op", "mi"):
            return n_lower
        return ""

    if isinstance(names, (list, tuple)):
        for i, n in enumerate(names):
            k = norm_to_key(str(n).lower())
            if k: mapping[i] = k
    elif isinstance(names, dict):
        for i, n in names.items():
            k = norm_to_key(str(n).lower())
            if k: mapping[int(i)] = k

    if not mapping:
        mapping = {i: DEFAULT_KEY_ORDER[i] for i in range(min(4, len(DEFAULT_KEY_ORDER)))}
    return mapping


def _choose_invoice_type(det, mapping_from_names: Dict[int, str]) -> str:
    import torch as _torch
    if det is None or det.shape[0] == 0:
        return "pc"
    confs = det[:, 4]
    idx = int(_torch.argmax(confs).item())
    cls_idx = int(det[idx, -1].item())
    name = mapping_from_names.get(cls_idx, "")
    if name in ("pc", "op", "mi"):
        return name
    fallback = ["pc", "op", "mi"]
    return fallback[cls_idx] if cls_idx < len(fallback) else "pc"

def _pad_box(x1, y1, x2, y2, W, H, l=0.0, t=0.0, r=0.0, b=0.0):
    """依比例對框做左右上下 padding；比例是相對於框寬/高。"""
    bw, bh = (x2 - x1), (y2 - y1)
    x1 = max(0, int(x1 - l * bw))
    y1 = max(0, int(y1 - t * bh))
    x2 = min(W - 1, int(x2 + r * bw))
    y2 = min(H - 1, int(y2 + b * bh))
    return x1, y1, x2, y2

def _crop(img_bgr, box: Tuple[int, int, int, int], out_path: str) -> bool:
    if cv2 is None or img_bgr is None or box is None:
        return False
    x1, y1, x2, y2 = box
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(int(x1), w - 1)); x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1)); y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img_bgr[y1:y2, x1:x2].copy()
    return cv2.imwrite(out_path, crop)


# ---------- 主流程 ----------
def detect_and_ocr(img_or_path: Any, crops_dir: Optional[str] = None, inv_type: str = "auto", **kwargs) -> Dict[str, Any]:
    """
    :param img_or_path: 影像路徑或 numpy array(BGR)
    :param crops_dir:   裁切輸出資料夾
    :param inv_type:    'auto' / 'pc' / 'op' / 'mi'
    :return: { type, num, date, sun, cash, crops: [{key,path,web_path}, ...] }
    """
    img_bgr = _load_bgr(img_or_path)
    if img_bgr is None:
        raise RuntimeError("載入圖片失敗（OpenCV 無法讀取）。")

    if crops_dir:
        _ensure_dir(crops_dir)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        crops_dir = os.path.join(here, "..", "uploads", "cropped")
        _ensure_dir(crops_dir)

    base_name = os.path.splitext(os.path.basename(str(img_or_path)))[0]
    nonce = uuid.uuid4().hex[:6]

    # 1) 判斷票種
    tr3 = _load_yolo_model(MODEL_PATHS["tr3"])
    class_map = _map_class_to_key(tr3)
    with torch.no_grad():
        # 放大到 960 比預設 640 更穩
        try:
            tr3.conf = float(os.environ.get("YOLO_CONF", 0.15))
            tr3.iou  = float(os.environ.get("YOLO_IOU", 0.45))
        except Exception:
            pass
        res_tr3 = tr3(img_bgr, size=960)
    det_tr3 = res_tr3.xyxy[0]
    inv = _choose_invoice_type(det_tr3, class_map) if inv_type == "auto" else inv_type.lower()
    if inv not in ("pc", "op", "mi"):
        inv = "pc"

    # 2) 欄位偵測
    model_inv = _load_yolo_model(MODEL_PATHS[inv])
    field_map = _map_class_to_key(model_inv)
    with torch.no_grad():
        # ↓ 降信心門檻 + 放大輸入尺寸
        try:
            model_inv.conf = float(os.environ.get("YOLO_CONF", 0.20))  # 你原本就有 0.20，可保留
            model_inv.iou  = float(os.environ.get("YOLO_IOU", 0.45))
        except Exception:
            pass
        res_fields = model_inv(img_bgr, size=1280)
    det = res_fields.xyxy[0]

    # 3) 裁切
    crops: List[Dict[str, str]] = []
    H, W = img_bgr.shape[:2]
    for row in det.tolist():
        x1, y1, x2, y2, conf, cls = row
        key = field_map.get(int(cls))
        if key not in ("num", "date", "sun", "cash"):
            continue

        # 👉 這裡新增
        if inv == "op" and key == "num":
            H, W = img_bgr.shape[:2]
            x1, y1, x2, y2 = _pad_box(int(x1), int(y1), int(x2), int(y2),
                                       W, H, l=0.05, t=0.02, r=0.45, b=0.02)
            
        if inv == "pc" and key in ("num", "cash"):
            x1, y1, x2, y2 = _pad_box(
                int(x1), int(y1), int(x2), int(y2), W, H,
                l=0.05, t=0.02, r=(0.30 if key == "num" else 0.20), b=0.02
            )

        out_file = f"{base_name}_{nonce}_{key}.jpg"
        out_path = os.path.join(crops_dir, out_file)
        if _crop(img_bgr, (int(x1), int(y1), int(x2), int(y2)), out_path):
            crops.append({"key": key, "path": out_file})

    # 4) OCR + 清洗（小圖）
    fields = ocr_fields_from_crops({c["key"]: os.path.join(crops_dir, c["path"]) for c in crops}, inv)

    # 4.1) 備援：小圖抓不到就用整頁錨點補
    missing = [k for k in ("num","date","sun","cash") if not fields.get(k)]
    need_fallback = (
        (inv in ("mi", "op") and (len(missing) >= 2 or "sun" in missing))
        or (inv == "pc" and any(k in missing for k in ("num", "cash", "date")))
    )
    if need_fallback:
        fb = fullpage_anchor_ocr(img_bgr, inv)
        for k in missing:
            if fb.get(k):
                fields[k] = fb[k]

    # 5) 裝上前端可用網址
    web_base = "/uploads/cropped"
    for c in crops:
        c["web_path"] = f"{web_base}/{c['path']}"

    return {
        "type": inv,
        "num":  fields.get("num", ""),
        "date": fields.get("date", ""),
        "sun":  fields.get("sun", ""),
        "cash": fields.get("cash", ""),
        "crops": crops,
    }


def visualize_yolo_results(model, img_path: str, save_path: str = "debug.jpg"):
    import cv2
    results = model(img_path)  # YOLO 預測
    
    # ✅ Debug 輸出 YOLO 偵測框
    print("\n[YOLO DEBUG]")
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        print(f"  Class={int(cls)} Conf={conf:.2f} BBox=({x1},{y1},{x2},{y2})")
    img = cv2.imread(img_path)
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{int(cls)} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imwrite(save_path, img)
    return save_path