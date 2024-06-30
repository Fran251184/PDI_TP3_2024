import cv2
import numpy as np


def detectar_lineas_carril(frame):
    height, width = frame.shape[:2]
    puntos_trapecio = np.array([
        [int(width * 0.05), height],
        [int(width * 0.95), height],
        [int(width * 0.55), int(height * 0.6)],
        [int(width * 0.45), int(height * 0.6)]
    ], dtype=np.int32)
    mascara = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mascara, [puntos_trapecio], 255)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    region_trapecio = cv2.bitwise_and(gray_frame, gray_frame, mask=mascara)
    median_val = np.median(region_trapecio[region_trapecio > 0])
    _, region_binarizada = cv2.threshold(region_trapecio, median_val + 80, 255, cv2.THRESH_BINARY)
    linesP = cv2.HoughLinesP(region_binarizada, 3, np.pi / 180, 100, None, 50, 200)
    return linesP

def dibujar_lineas(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined_image


def grabar_video_entrada(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error al abrir el archivo de video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Propiedades del video - Ancho: {width}, Alto: {height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame o fin del video")
            break

        linesP = detectar_lineas_carril(frame)
        frame_con_lineas = dibujar_lineas(frame, linesP)

        cv2.imshow('Video Processing', frame_con_lineas)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        out.write(frame_con_lineas)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Procesados {frame_count} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path1 = 'ruta_1.mp4'
    output_video_path1 = 'ruta_1_output.mp4'
    print(f"Procesando {input_video_path1}...")
    grabar_video_entrada(input_video_path1, output_video_path1)
    
    input_video_path2 = 'ruta_2.mp4'
    output_video_path2 = 'ruta_2_output.mp4'
    print(f"Procesando {input_video_path2}...")
    grabar_video_entrada(input_video_path2, output_video_path2)

    print("Procesamiento completado.")
