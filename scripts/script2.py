from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time

# Inicializa câmera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)

# Carrega modelo YOLO
print("🔄 Carregando modelo YOLO...")
model = YOLO("../models/best_MediTrack_IESTI05.pt")
print("✅ Modelo carregado!")

# Confiança mínima
CONF_MIN = 0.7

print(f"📸 Iniciando ciclo de captura e detecção (confiança mínima = {CONF_MIN})...")
time.sleep(1)

try:
    while True:
        # --- ETAPA 1: Captura ---
        picam2.start()
        time.sleep(0.3)  # tempo para estabilizar a câmera
        frame = picam2.capture_array()
        picam2.stop()

        # --- ETAPA 2: Processamento YOLO ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=frame_rgb, verbose=False)

        # --- ETAPA 3: Filtra e mostra resultados ---
        detected_any = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Aplica filtro de confiança mínima
                if conf >= CONF_MIN:
                    detected_any = True
                    print(f"🧠 Objeto detectado → classe {cls}, confiança {conf:.2f}")
                else:
                    print(f"⚠️ Detecção ignorada (confiança {conf:.2f} < {CONF_MIN})")

        if not detected_any:
            print("🔍 Nenhum remédio detectado com confiança suficiente.")

        # Espera antes do próximo ciclo
        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Encerrado pelo usuário.")
