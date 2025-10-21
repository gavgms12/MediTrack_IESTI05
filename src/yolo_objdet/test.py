import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from picamera2 import Picamera2
import fastapi
from fastapi.responses import StreamingResponse
import threading
import io
import os

# Caminho do modelo treinado
model_path = "models/best_MediTrack_IESTI05.pt"
model = YOLO(model_path)

# Inicializando a API FastAPI
app = fastapi.FastAPI()

# Configuração da câmera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (320, 320)}))  # Resolução reduzida
picam2.start()

# Função para capturar e processar o frame
def capture_frame():
    """Captura um frame da câmera e executa a inferência"""
    frame = picam2.capture_array()
    return process_frame(frame)

# Função para processar o frame e rodar o modelo YOLO
def process_frame(frame):
    """Roda o modelo YOLO no frame capturado e retorna a imagem processada"""
    # Convertendo o frame de BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Inferência YOLO
    results = model.predict(pil_image, imgsz=320, conf=0.5, iou=0.3)  # Imagem 320x320
    result = results[0]
    
    # Gerando o frame com as detecções
    im_bgr = result.plot()
    im_rgb = im_bgr[..., ::-1]  # Convertendo de BGR para RGB
    return im_rgb

# Função para gerar o stream de vídeo
def gen():
    """Gera frames para o stream em tempo real"""
    while True:
        frame = capture_frame()
        # Codificando o frame para JPEG
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n\r\n')

# Endpoint para o feed de vídeo
@app.get('/video_feed')
async def video_feed():
    """Retorna o stream de vídeo contínuo"""
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# Função para monitorar a temperatura da Raspberry Pi
def monitor_temperature():
    """Monitora a temperatura da Raspberry Pi"""
    while True:
        temp = os.popen("vcgencmd measure_temp").readline()
        temp = float(temp.replace("temp=", "").replace("'C\n", ""))
        if temp > 70:  # Limite de temperatura para a Raspberry Pi
            print(f"Temperatura alta: {temp}°C - Aguardando resfriamento...")
            time.sleep(5)  # Pausa por 5 segundos se a temperatura estiver alta
        else:
            time.sleep(1)

# Thread para monitorar a temperatura enquanto o servidor está rodando
temperature_thread = threading.Thread(target=monitor_temperature, daemon=True)
temperature_thread.start()

# Executando o servidor FastAPI
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
