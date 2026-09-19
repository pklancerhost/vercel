from flask import Flask

try:
    from hostinger_app.app import app as app
except Exception:
    from zip_check.app import app as app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
