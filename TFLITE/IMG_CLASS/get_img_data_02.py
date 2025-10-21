from flask import Flask, Response, render_template_string, request, redirect, url_for, send_from_directory
from picamera2 import Picamera2
import io
import threading
import time
import os
import signal

app = Flask(__name__)

base_dir = "dataset"
os.makedirs(base_dir, exist_ok=True)

picam2 = None
frame = None
frame_lock = threading.Lock()
capture_counts = {}
current_label = None
last_captured_filename = None
shutdown_event = threading.Event()

def initialize_camera():
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

def get_frame():
    global frame
    while not shutdown_event.is_set():
        stream = io.BytesIO()
        picam2.capture_file(stream, format='jpeg')
        with frame_lock:
            frame = stream.getvalue()
        time.sleep(0.1)

def generate_frames():
    while not shutdown_event.is_set():
        with frame_lock:
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

def shutdown_server():
    shutdown_event.set()
    if picam2:
        picam2.stop()
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGINT)

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_label
    if request.method == 'POST':
        current_label = request.form['label']
        if current_label not in capture_counts:
            capture_counts[current_label] = 0
        os.makedirs(os.path.join(base_dir, current_label), exist_ok=True)
        return redirect(url_for('capture_page'))
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Capture - Label Entry</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { background-color: #f8f9fa; }
                .container { max-width: 600px; margin-top: 100px; }
                .card { border-radius: 16px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="container text-center">
                <div class="card p-4">
                    <h2 class="mb-4">Dataset Capture</h2>
                    <form method="post" class="d-flex justify-content-center">
                        <input type="text" name="label" class="form-control w-75 me-2" placeholder="Digite o nome do rótulo" required>
                        <button type="submit" class="btn btn-primary">Iniciar</button>
                    </form>
                </div>
            </div>
        </body>
        </html>
    ''')

@app.route('/capture')
def capture_page():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Capture</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { background-color: #f8f9fa; }
                .content { max-width: 900px; margin: 40px auto; text-align: center; }
                .video-box, .image-box {
                    border-radius: 12px;
                    background: white;
                    padding: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    margin-bottom: 25px;
                }
                img { border-radius: 8px; }
            </style>
            <script>
                var shutdownInitiated = false;
                function checkShutdown() {
                    if (!shutdownInitiated) {
                        fetch('/check_shutdown')
                            .then(response => response.json())
                            .then(data => {
                                if (data.shutdown) {
                                    shutdownInitiated = true;
                                    document.getElementById('video-feed').src = '';
                                    document.getElementById('shutdown-message').style.display = 'block';
                                }
                            });
                    }
                }
                setInterval(checkShutdown, 1000);
            </script>
        </head>
        <body>
            <div class="content">
                <h2 class="mb-3 text-primary">Captura de Dataset</h2>
                <p><strong>Rótulo atual:</strong> {{ label }}</p>
                <p><strong>Imagens capturadas:</strong> {{ capture_count }}</p>

                <div class="video-box">
                    <h5>Visualização da Câmera</h5>
                    <img id="video-feed" src="{{ url_for('video_feed') }}" width="640" height="480" class="img-fluid"/>
                </div>

                {% if last_image %}
                <div class="image-box">
                    <h5>Última imagem capturada:</h5>
                    <img src="{{ url_for('serve_image', label=label, filename=last_image) }}" width="320" class="img-thumbnail mt-2">
                </div>
                {% endif %}

                <div id="shutdown-message" class="alert alert-danger" style="display:none;">
                    O processo de captura foi encerrado. Você pode fechar esta janela.
                </div>

                <div class="d-flex justify-content-center gap-3">
                    <form action="/capture_image" method="post">
                        <button type="submit" class="btn btn-success">Capturar Imagem</button>
                    </form>
                    <form action="/stop" method="post">
                        <button type="submit" class="btn btn-danger">Parar Captura</button>
                    </form>
                    <form action="/" method="get">
                        <button type="submit" class="btn btn-warning">Trocar Rótulo</button>
                    </form>
                </div>
            </div>
        </body>
        </html>
    ''', label=current_label,
         capture_count=capture_counts.get(current_label, 0),
         last_image=last_captured_filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global capture_counts, last_captured_filename
    if current_label and not shutdown_event.is_set():
        capture_counts[current_label] += 1
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"image_{timestamp}.jpg"
        full_path = os.path.join(base_dir, current_label, filename)
        picam2.capture_file(full_path)
        last_captured_filename = filename
    return redirect(url_for('capture_page'))

@app.route('/images/<label>/<filename>')
def serve_image(label, filename):
    return send_from_directory(os.path.join(base_dir, label), filename)

@app.route('/stop', methods=['POST'])
def stop():
    summary = render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Capture - Stopped</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light text-center mt-5">
            <div class="container">
                <div class="card p-4 mx-auto" style="max-width:600px;">
                    <h2 class="text-danger mb-3">Captura Encerrada</h2>
                    <p>O processo foi finalizado. Você pode fechar esta janela.</p>
                    <h4 class="mt-4">Resumo das Capturas:</h4>
                    <ul class="list-group">
                    {% for label, count in capture_counts.items() %}
                        <li class="list-group-item d-flex justify-content-between">
                            <span>{{ label }}</span>
                            <span class="badge bg-primary rounded-pill">{{ count }} imagens</span>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
            </div>
        </body>
        </html>
    ''', capture_counts=capture_counts)
    threading.Thread(target=shutdown_server).start()
    return summary

@app.route('/check_shutdown')
def check_shutdown():
    return {'shutdown': shutdown_event.is_set()}

if __name__ == '__main__':
    initialize_camera()
    threading.Thread(target=get_frame, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
