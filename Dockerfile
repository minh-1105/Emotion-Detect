FROM python:3.12

LABEL AUTHOR="minh"

WORKDIR /app

#system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

#reqirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#app
COPY app/api.py ./app/api.py
#model, framerwork
COPY artifacts/models/Resnet50 ./artifacts/models/Resnet50
COPY artifacts/models/EfficientNetB0 ./artifacts/models/EfficientNetB0
COPY artifacts/models/mobilenetv2 ./artifacts/models/mobilenetv2
COPY artifacts/models/Resemotenet ./artifacts/models/Resemotenet
COPY artifacts/models/mediapipe ./artifacts/models/mediapipe

EXPOSE 8888

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8888"]
