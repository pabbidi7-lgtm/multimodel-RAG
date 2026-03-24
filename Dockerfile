FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pymilvus \
    PyMuPDF \
    requests \
    python-dotenv

COPY pipeline.py /app/pipeline.py

CMD ["python", "pipeline.py"]