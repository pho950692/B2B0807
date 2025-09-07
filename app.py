# -*- coding: utf-8 -*-
# 啟動 Flask 的入口：載入 core_app 以建立 app，再匯入各路由檔（yr 等）
from core_app import app
import auth_routes      # noqa: F401
import comp             # noqa: F401
import inv              # noqa: F401
import reinv            # noqa: F401
import pre              # noqa: F401
import yr               # ✅ 路由、上傳、結果頁都在這支

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

# （可選）啟動時列出路由，方便除錯
print("\n== Routes ==")
for r in app.url_map.iter_rules():
    print(f"{r.endpoint:25s} -> {r}")
