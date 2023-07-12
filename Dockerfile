FROM python:3.9.5

ADD . /app

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgl1

RUN apt-get update && apt-get install -y poppler-utils

RUN pip install openvino==2023.0.1

CMD ["python", "app.py"]