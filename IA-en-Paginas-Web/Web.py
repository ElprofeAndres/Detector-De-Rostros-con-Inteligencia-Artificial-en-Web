# Importamos librerias
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

# Creamos nuestra funcion de dibujo
mp_dibujo = mp.solutions.drawing_utils
conf_dibu = mp_dibujo.DrawingSpec(thickness=1, circle_radius=1)

# Creamos un objeto donde almacenaremos la malla facial
mp_malla_facial = mp.solutions.face_mesh
malla_facial = mp_malla_facial.FaceMesh(max_num_faces=1)

# Realizamos la Videocaptura
cap = cv2.VideoCapture(0)  # Asegurado para la cámara predeterminada al dispositivo.

# Creamos la app
app = Flask(__name__)

# Mostramos el video en RT
def gen_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = malla_facial.process(frameRGB)
        if resultados.multi_face_landmarks:
            for rostros in resultados.multi_face_landmarks:
                mp_dibujo.draw_landmarks(frame, rostros, mp_malla_facial.FACEMESH_TESSELATION, conf_dibu, conf_dibu)
        suc, encode = cv2.imencode('.jpg', frame)
        frame = encode.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Ruta de aplicacion 'principal'
@app.route('/')
def index():
    return render_template('Index.html')

# Ruta del video
@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutamos la app
if __name__ == "__main__":
    app.run(debug=True)