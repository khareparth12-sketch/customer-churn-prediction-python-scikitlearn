FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 10000

CMD ["bash", "-c", "uvicorn api.app:app --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run dashboard/app.py --server.port 10000 --server.address 0.0.0.0"]