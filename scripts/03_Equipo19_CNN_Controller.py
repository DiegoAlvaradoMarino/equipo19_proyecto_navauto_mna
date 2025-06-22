import numpy as np
import cv2
from controller import Robot, Camera, GPS
from vehicle import Car, Driver
from keras.models import load_model
from keras.optimizers import Adam

# === Constantes ===
THRESHOLD_DISTANCE_CAR = 6.5  # Distancia mínima (en metros) para detener el coche frente a otro vehículo
CAR_SPEED = 10                # Velocidad estándar del vehículo en km/h

# === Controlador principal del vehículo con red neuronal convolucional ===
class CNNController:
    MAX_ANGLE = 0.28  # Ángulo máximo que puede tomar el volante (en radianes)

    def __init__(self):
        """Inicializa el robot, dispositivos y parámetros de control."""
        self.robot = Car()
        self.driver = Driver()
        self.timestep = int(self.robot.getBasicTimeStep())  # Paso de simulación definido por Webots

        # Inicialización de sensores
        self.camera = self._init_device("camera")
        self.lidar = self._init_lidar("lidar")
        self.display = self.robot.getDevice("display")

        self.angle = 0.0  # Ángulo inicial del volante
        self.speed = 25.0  # Velocidad inicial (se puede ajustar por lógica)

    def _init_device(self, device_name):
        """Inicializa un dispositivo Webots por nombre."""
        device = self.robot.getDevice(device_name)
        device.enable(self.timestep)
        return device

    def _init_lidar(self, device_name):
        """Inicializa el lidar y activa la nube de puntos."""
        lidar = self.robot.getDevice(device_name)
        lidar.enable(self.timestep)
        lidar.enablePointCloud()
        return lidar

    def update_display(self):
        """Dibuja en pantalla la velocidad, ángulo de dirección y etiquetas."""
        speed = self.driver.getCurrentSpeed()
        steering_angle = self.driver.getSteeringAngle()

        # Fondo del display
        self.display.setColor(0x202020)
        self.display.fillRectangle(0, 0, self.display.getWidth(), self.display.getHeight())

        # Etiquetas
        self.display.setColor(0xFFD700)
        self.display.drawText("Equipo 19", 0, 0)
        self.display.drawText("Navegacion Autonoma", 100, 0)

        # Colores y estilos de elementos visuales
        box_color = 0x404040
        border_color = 0xFF4500
        label_color = 0x00BFFF
        value_color = 0xFFFFFF

        # Cajas y bordes
        self.display.setColor(box_color)
        self.display.fillRectangle(5, 25, 290, 25)
        self.display.fillRectangle(5, 55, 290, 25)

        self.display.setColor(border_color)
        self.display.drawRectangle(5, 25, 290, 25)
        self.display.drawRectangle(5, 55, 290, 25)

        # Texto de etiquetas y valores
        self.display.setColor(label_color)
        self.display.drawText("Speed:", 10, 30)
        self.display.drawText("Angle:", 10, 60)

        self.display.setColor(value_color)
        self.display.drawText(f"{speed:.2f} km/h", 150, 30)
        self.display.drawText(f"{steering_angle:.5f} rad", 150, 60)

    def set_steering_angle(self, value):
        """Establece el ángulo de dirección (con zona muerta para evitar ruido)."""
        DEAD_ZONE = 0.06
        value = value if abs(value) > DEAD_ZONE else 0.0
        self.angle = self.MAX_ANGLE * value  # Normalización del valor [-1, 1] al rango físico del coche

    def set_speed(self, kmh):
        """Establece la velocidad de crucero del vehículo (km/h)."""
        self.speed = kmh
        self.driver.setCruisingSpeed(self.speed)

    def update(self):
        """Actualiza la lógica de conducción y el contenido del display."""
        self.update_display()
        self.driver.setSteeringAngle(self.angle)
        self.driver.setCruisingSpeed(self.speed)

    def get_image(self):
        """Obtiene una imagen desde la cámara, la recorta y redimensiona para el modelo."""
        raw_image = self.camera.getImage()

        # Convertir buffer en arreglo NumPy con forma (H, W, 4) por canal alpha
        image = np.frombuffer(raw_image, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))

        # Preprocesamiento visual (recorte + resize)
        image = cv2.resize(image, (200, 66))
        image = image[35:, :, :]  # Eliminar parte superior (cielo u horizonte)
        image = cv2.resize(image, (200, 66))  # Asegurar tamaño estándar requerido por el modelo
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Convertir de BGRA a BGR
        return image

    def get_lid_ranges(self):
        """Procesa los datos del lidar, calcula métricas y clasifica el tipo de obstáculo."""
        range_image = self.lidar.getRangeImage()

        # Filtrar valores infinitos (sin retorno válido)
        ranges = [val for val in range_image if not np.isinf(val)]
        num_lasers = len(ranges)
        mean_range = np.mean(ranges)
        min_range = min(ranges) if ranges else float('inf')

        # Clasificación basada en cantidad de puntos detectados
        detection = None
        if num_lasers == 0:
            print("Detected: None")
        elif num_lasers < 150:
            detection = "Pedestrian"
            print("Detected: Pedestrian")
        else:
            detection = "Car"
            print("Detected: Car")

        print(f'Num Lasers: {num_lasers}')
        return mean_range, num_lasers, min_range, detection

# === Bucle principal ===
def main_loop(car, model):
    """Bucle de ejecución del vehículo: obtiene imagen, predice, ajusta movimiento y evalúa obstáculos."""
    try:
        TIMER = 30     # Intervalo para realizar inferencia (controla frecuencia de evaluación)
        COUNTER = 0
        predicted_steering_angle = 0.0

        while car.robot.step() != -1:
            if COUNTER == TIMER:
                image = car.get_image()
                preprocessed_image = np.array([image])  # Añadir batch dimension (1, H, W, C)
                predicted_steering_angle = model.predict(preprocessed_image)[0][0]
                print(f"Predicted steering angle: {predicted_steering_angle}")

                car.set_steering_angle(predicted_steering_angle)

                _, _, min_range, detection = car.get_lid_ranges()
                if detection == "Pedestrian": # Detección de Persona
                    car.set_speed(0)
                elif detection == "Car": # Detección de Carro
                    car.set_speed(0 if min_range < THRESHOLD_DISTANCE_CAR else CAR_SPEED)
                else:
                    car.set_speed(CAR_SPEED)

                car.update()
                print(f"Vehicle Speed: {car.speed} km/h, Steering Angle: {car.angle} rad")
                COUNTER = 0

            COUNTER += 1

    finally:
        print("Exiting the main loop.")

# === Punto de entrada principal ===
if __name__ == "__main__":
    car = CNNController()
    model_path = r'C:\Users\Diego Alvarado\Documents\Trimestre 5\navegacion_autonoma\Proyecto\equipo19_proyecto_navauto_mna\models\behavioral_cloning_v5_05_x10_ogcnn.h5'
    
    # Cargar el modelo previamente entrenado (sin recompilarlo)
    model = load_model(model_path, compile=False)
    
    # Compilar con función de pérdida MSE y optimizador Adam
    model.compile(Adam(learning_rate=0.001), loss='mse')

    # Ejecutar bucle principal
    main_loop(car, model)
