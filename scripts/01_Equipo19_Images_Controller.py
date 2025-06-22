from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

# === CONFIGURACIONES ===
BASE_DIR = "images_data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CSV_FILE = os.path.join(BASE_DIR, "images_records.csv")
SAVE_INTERVAL = 0.5
IMAGE_SIZE = (128, 64)  # Debe coincidir con Webots

# === FUNCIONES AUXILIARES ===
def get_image(camera):
    raw_image = camera.getImage()
    return np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )

def display_image(display, image):
    image_rgb = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# === CONTROL DE VELOCIDAD Y DIRECCIÓN ===
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 10

def set_speed(kmh):
    global speed
    speed = kmh

def set_steering_angle(wheel_angle):
    global angle, steering_angle
    delta = wheel_angle - steering_angle
    if delta > 0.1:
        wheel_angle = steering_angle + 0.1
    elif delta < -0.1:
        wheel_angle = steering_angle - 0.1
    wheel_angle = max(min(wheel_angle, 0.5), -0.5)
    steering_angle = wheel_angle
    angle = wheel_angle

def change_steer_angle(inc):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if -25.0 <= new_manual_steering <= 25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)

# === LOOP PRINCIPAL ===
def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            f.write("Image Name\tSteering Angle\n")

    last_save_time = time.time()
    save_counter = 0  # Contador para sufijo único en los nombres

    date_prefix = datetime.now().strftime("DAM-%Y-%m-%d_%H-%M")

    while robot.step() != -1:
        full_image = get_image(camera)

        # Mostrar imagen (escala de grises)
        grey_image = cv2.cvtColor(full_image, cv2.COLOR_RGBA2GRAY)
        display_image(display_img, grey_image)

        key = keyboard.getKey()
        if key == keyboard.UP:
            set_speed(speed + 5.0)
        elif key == keyboard.DOWN:
            set_speed(speed - 5.0)
        elif key == keyboard.RIGHT:
            change_steer_angle(+1)
        elif key == keyboard.LEFT:
            change_steer_angle(-1)

        # Guardado periódico
        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL:
            filename = f"{date_prefix}-{save_counter}.png"
            filepath = os.path.join(IMAGE_DIR, filename)

            # Guardar imagen
            image_bgr = cv2.cvtColor(full_image, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(filepath, image_bgr)

            # Guardar en CSV con \t como separador y ruta relativa
            with open(CSV_FILE, mode='a', newline='') as f:
                f.write(f"images\\{filename}\t{angle:.9f}\n")

            print(f"Saved: {filename}, angle: {angle:.9f}")
            last_save_time = current_time
            save_counter += 1

        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()
