#!/usr/bin/env python3
"""
meds_monitor_rt.py
Real-time medication recognition on Raspberry Pi W2 using Picamera2 + Ultralytics YOLO.
Logs events to Google Sheets; optional: upload images to Google Drive and save link to sheet.

Ajuste:
 - MODEL_PATH: caminho para seu modelo .pt
 - SHEET_NAME: nome da planilha Google (ou URL)
 - CREDENTIALS_JSON: arquivo da service account
 - INTERVAL_SECONDS: intervalo entre capturas (5s pedido)
 - DEBOUNCE_SECONDS: 30 minutos = 1800s
 - CLASS_NAMES: se quiser sobrescrever; caso contrário usa model.names
 - SCHEDULE: optional dict com horários esperados (em "HH:MM"), e tolerância em minutos
"""

import time
import io
import os
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from ultralytics import YOLO
from picamera2 import Picamera2
import threading

# Google libs
import gspread
from google.oauth2.service_account import Credentials

# Optional Drive upload
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.auth.transport.requests import Request

# ----------------- CONFIG -----------------
MODEL_PATH = "../models/best_MediTrack_IESTI05.pt"
CREDENTIALS_JSON = "credentials.json"   # sua service account JSON
SHEET_NAME = "MediTrack"         # nome da planilha criada e compartilhada com a service account
DRIVE_FOLDER_ID = "1lvD1lhVAI9np996RR6Ohi6XEwBV7RR_w"  # coloca folder id se quiser salvar imagens em pasta do Drive (opcional). Ex: "1AbC..."
INTERVAL_SECONDS = 5
DEBOUNCE_SECONDS = 30 * 60  # 30 minutos
CONF_THRESHOLD = 0.5
IOU = 0.3
IMG_SAVE_LOCAL = "./captured_images"   # pasta local para salvar imagens (opcional)
UPLOAD_TO_DRIVE = True  # True para enviar a imagem ao Drive e salvar link na planilha (requer DRIVE_FOLDER_ID possivelmente)
# ------------------------------------------

if not os.path.exists(IMG_SAVE_LOCAL):
    os.makedirs(IMG_SAVE_LOCAL, exist_ok=True)

# Optional schedule: times per medication (strings "HH:MM" for expected times) and tolerance in minutes
SCHEDULE = {
    "cefalexina": ["07:00", "19:00"],
    "cloridrato_de_ambroxol": ["09:00", "21:00"],
    "sany_d": ["12:00"],    
}
SCHEDULE_TOLERANCE_MIN = 60  # +/- tolerance in minutes to consider 'on_time' (ajuste conforme necessidade)

# ----------------- INITIALIZE MODEL -----------------
print("Carregando modelo YOLO...")
model = YOLO(MODEL_PATH, task="detect", verbose=True)  # pode ajustar verbose

# Obter nomes de classes do modelo
try:
    MODEL_CLASS_NAMES = model.names  # Tentando pegar os nomes das classes
    names = MODEL_CLASS_NAMES or {}  # Se não houver nomes, usa um dicionário vazio
except Exception as e:
    print(f"Erro ao obter nomes das classes: {e}")
    names = {}  # Definindo como um dicionário vazio caso o acesso falhe

if names:
    print("Classes do modelo:", names)
else:
    print("Nenhum nome de classe encontrado, usando dicionário vazio.")

# ----------------- GOOGLE SHEETS AUTH -----------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

creds = None
if os.path.exists(CREDENTIALS_JSON):
    creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)
    try:
        sheet = gc.open(SHEET_NAME).sheet1
    except Exception as e:
        print("Erro abrindo planilha:", e)
        print("Tentando criar planilha...")
        sh = gc.create(SHEET_NAME)
        sh.share(None, perm_type='anyone', role='reader')
        sheet = sh.sheet1
else:
    sheet = None
    print("Arquivo de credenciais não encontrado. A gravação no Google Sheets ficará desabilitada.")

# Optional Drive service for upload
drive_service = None
if UPLOAD_TO_DRIVE and creds:
    drive_service = build('drive', 'v3', credentials=creds)

# Ensure header in sheet
if sheet:
    try:
        header = sheet.row_values(1)
        if not header:
            sheet.insert_row(["timestamp_utc", "timestamp_local", "med_name", "class_id", "confidence", "on_time", "schedule_expected", "image_url"], 1)
    except Exception as e:
        print("Erro ao garantir cabeçalho na sheet:", e)

# ----------------- DEBOUNCE STORAGE -----------------
# Guarda timestamp (unix) do último evento gravado por med_name
last_logged = {}

def is_within_schedule(med_name, ts_dt):
    """Retorna (bool_on_time, expected_time_str_or_empty)"""
    if med_name not in SCHEDULE:
        return (None, "")   # sem schedule configurada
    times = SCHEDULE[med_name]
    for tstr in times:
        hh, mm = map(int, tstr.split(":"))
        expected = ts_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_min = abs((ts_dt - expected).total_seconds()) / 60.0
        if delta_min <= SCHEDULE_TOLERANCE_MIN:
            return (True, tstr)
    return (False, ",".join(times))

def upload_image_to_drive(filepath, filename=None, folder_id=None):
    """Faz upload para Google Drive e retorna link 'anyoneWithLink' (requer drive_service)."""
    if drive_service is None:
        return ""
    filename = filename or os.path.basename(filepath)
    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    media = MediaIoBaseUpload(open(filepath, "rb"), mimetype='image/jpeg')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    # Tornar compartilhável publicamente (anyone with link)
    try:
        drive_service.permissions().create(fileId=file_id, body={'role': 'reader', 'type': 'anyone'}).execute()
        link = f"https://drive.google.com/uc?id={file_id}"
    except Exception as e:
        print("Erro definindo permissão do arquivo no Drive:", e)
        link = ""
    return link

def log_event_to_sheet(timestamp_utc, timestamp_local, med_name, class_id, conf, on_time_flag, schedule_expected, image_url=""):
    row = [timestamp_utc.isoformat(), timestamp_local.isoformat(), med_name, int(class_id), float(conf), str(on_time_flag), schedule_expected, image_url]
    if sheet:
        try:
            sheet.append_row(row)
        except Exception as e:
            print("Erro ao gravar na sheet:", e)
    else:
        print("Planilha não configurada — evento:", row)

# ----------------- CAMERA SETUP -----------------
print("Configurando Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 640)})  # 640 good balance
picam2.configure(config)
picam2.start()
time.sleep(2)  # tempo de aquecimento

# Warm-up model (uma predição dummy para evitar delays maiores no primeiro frame)
try:
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model.predict(dummy, imgsz=640, conf=0.1, iou=0.3, verbose=False)
    print("Modelo aquecido.")
except Exception as e:
    print("Aviso: problema ao aquecer modelo:", e)

print("Iniciando loop principal. CTRL+C para parar.")
try:
    while True:
        start = time.time()
        # Capture array (RGB)
        frame = picam2.capture_array()
        if frame.shape[2] == 4:  # se vier com 4 canais
            frame = frame[:, :, :3]  # descarta o canal extra (alfa)
        # Execute inferência
        try:
            results = model.predict(source=frame, save=False, imgsz=640, conf=CONF_THRESHOLD, iou=IOU, verbose=False)
            res = results[0]
        except Exception as e:
            print("Erro durante inferência:", e)
            res = None

        if res is not None:
            # Dependendo da versão, a API pode variar. Vamos extrair com verificação defensiva.
            boxes = getattr(res, "boxes", None)
            names = names or {}
            detected_this_frame = []
            if boxes is not None:
                # boxes.cls e boxes.conf may be tensors -> convert to numpy
                cls_arr = []
                conf_arr = []
                try:
                    cls_arr = boxes.cls.cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy()
                except Exception:
                    # fallback if attributes different
                    try:
                        for b in boxes:
                            cls_arr.append(float(b.cls))
                            conf_arr.append(float(b.conf))
                    except Exception:
                        cls_arr = []
                        conf_arr = []

                for i, cls_id in enumerate(cls_arr):
                    conf = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                    if conf < CONF_THRESHOLD:
                        continue
                    cls_id_int = int(cls_id)
                    med_name = names.get(cls_id_int, str(cls_id_int))
                    detected_this_frame.append((med_name, cls_id_int, conf))

            # deduplicate by med_name within same frame (several boxes same med)
            unique_detected = {}
            for med_name, cls_id_int, conf in detected_this_frame:
                # keep highest confidence per med in this frame
                if med_name not in unique_detected or conf > unique_detected[med_name][1]:
                    unique_detected[med_name] = (cls_id_int, conf)

            # If any detected, log with debounce logic
            for med_name, (cls_id_int, conf) in unique_detected.items():
                now = datetime.utcnow()
                now_local = datetime.now()  # local timezone of the Pi
                last_ts = last_logged.get(med_name)
                allowed = False
                if last_ts is None:
                    allowed = True
                else:
                    elapsed = (time.time() - last_ts)
                    if elapsed >= DEBOUNCE_SECONDS:
                        allowed = True
                if allowed:
                    # schedule check
                    on_time_flag, schedule_expected = is_within_schedule(med_name, now_local)
                    image_url = ""
                    # Save image locally and optionally upload
                    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                    fname = f"{med_name}_{timestamp_str}.jpg"
                    local_path = os.path.join(IMG_SAVE_LOCAL, fname)
                    try:
                        # Save PIL image from frame (frame is numpy RGB)
                        pil = Image.fromarray(frame)
                        pil.save(local_path, format="JPEG", quality=85)
                    except Exception as e:
                        print("Erro salvando imagem local:", e)
                        local_path = None

                    if UPLOAD_TO_DRIVE and local_path and drive_service:
                        try:
                            image_url = upload_image_to_drive(local_path, filename=fname, folder_id=DRIVE_FOLDER_ID)
                        except Exception as e:
                            print("Erro upload Drive:", e)
                            image_url = ""

                    # Log to sheet
                    try:
                        log_event_to_sheet(now, now_local, med_name, cls_id_int, conf, on_time_flag, schedule_expected, image_url)
                        print(f"[{now_local.isoformat()}] Registrado: {med_name} conf={conf:.2f} on_time={on_time_flag} image={image_url or local_path}")
                    except Exception as e:
                        print("Erro ao registrar evento:", e)
                    # Atualiza debounce
                    last_logged[med_name] = time.time()
                else:
                    # Debounced
                    next_allowed = last_logged[med_name] + DEBOUNCE_SECONDS
                    nxt_dt = datetime.utcfromtimestamp(next_allowed)
                    print(f"{med_name} detectado, mas debounced (permitido novamente: {nxt_dt.isoformat()} UTC).")

        # Ajusta espera para manter ~INTERVAL_SECONDS entre capturas
        elapsed_total = time.time() - start
        sleep_for = INTERVAL_SECONDS - elapsed_total
        if sleep_for > 0:
            time.sleep(sleep_for)

except KeyboardInterrupt:
    print("Interrompido pelo usuário. Encerrando.")
finally:
    try:
        picam2.stop()
    except:
        pass
