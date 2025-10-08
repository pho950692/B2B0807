from core_app import app
import admin_routes
import auth_routes
import comp
import inv
import reinv
import pre
import yr  # ✅ 改用 ocr，別再 import yocr
def print_routes():
    print("\n== Routes ==")
    for r in app.url_map.iter_rules():
        print(f"{r.endpoint:25s} -> {r}")

if __name__ == "__main__":
    print_routes()
    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",   # 🔴 讓 Flask 對外可達（包含 ZeroTier）
        port=5000
    )
