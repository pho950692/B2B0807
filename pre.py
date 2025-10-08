# pre_routes.py
from core_app import app
from flask import render_template, request, redirect, url_for, flash, session

@app.route("/preferences", methods=["GET", "POST"], endpoint="preferences")

def preferences():
    if "user_id" not in session:
        flash("請先登入")
        return redirect(url_for("login"))

    user_email = None
    from db import get_db
    conn = get_db()
    if conn:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute("SELECT mail FROM users WHERE id=%s", (session["user_id"],))
            row = cur.fetchone()
            if row:
                user_email = (row.get("mail") or "").strip()
        finally:
            try: cur.close(); conn.close()
            except Exception: pass

    if request.method == "POST":
        # ...existing code...
        flash('偏好設定已更新')
        return redirect(url_for('preferences'))

    return render_template("preferences.html", user_email=user_email)
