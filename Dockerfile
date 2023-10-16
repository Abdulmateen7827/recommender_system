FROM python:3.9.15-slim

WORKDIR /app

COPY . /app/

RUN pip install -r /app/requirements.txt && \
	rm requirements.txt

EXPOSE 80

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
