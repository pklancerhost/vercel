HOSTINGER PYTHON DEPLOYMENT

This package is intended for a Hostinger VPS or a Hostinger Python application
that supports PyTorch and long-running WSGI applications. Shared hosting may
reject PyTorch because of memory, CPU, or package restrictions.

1. Upload and extract this folder.
2. Create a Python virtual environment for the application.
3. Install requirements.txt in that environment.
4. Set the application startup file to passenger_wsgi.py.
5. Set the application entry point to application.
6. Restart the Python application.

For a VPS, use a systemd service and Nginx reverse proxy. The app listens on
127.0.0.1:8000 when run directly. The first startup downloads the TorchVision
Faster R-CNN model, so allow several GB of disk space and enough RAM.

Test endpoint: GET /
Prediction endpoint: POST /predict with a multipart field named image