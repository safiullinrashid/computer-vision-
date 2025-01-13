import cv2
import numpy as np


# Функция для обработки кадра
def process_frame(frame, color='yellow'):
    # Преобразуем кадр в HSV для детекции по цвету
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Задаем диапазон цвета для детекции
    if color == 'yellow':
        # Диапазон для желтого цвета
        lower_color = np.array([20, 70, 70])  # Нижняя граница HSV для желтого
        upper_color = np.array([30, 255, 255])  # Верхняя граница HSV для желтого
    elif color == 'gray':
        # Диапазон для серого цвета
        lower_color = np.array([0, 0, 0])  # Нижняя граница HSV для черного
        upper_color = np.array([179, 150, 50])  # Верхняя граница HSV для черного

    # Создаем маску для выделения объекта
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Применяем морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Удаление шума
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Заполнение отверстий

    # Находим контуры объекта
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры найдены
    if contours:
        # Выбираем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)

        # Рисуем контур на кадре
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        # Вычисляем центр масс контура
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Рисуем центр масс
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            # Выводим координаты центра масс
            cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Если объект не найден
        cv2.putText(frame, "Object not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


# Часть 1: Обработка видеофайла "catball.mp4"
def process_video_file():
    video_path = r"C:\Users\User\Downloads\Telegram Desktop\catball.mp4"

    # Загрузка видеофайла
    cap = cv2.VideoCapture(video_path)

    # Проверка, открылось ли видео
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        return

    # Обработка видео и отображение кадров
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра (желтый цвет)
        processed_frame = process_frame(frame, color='yellow')

        # Отображение кадра
        cv2.imshow('Processed Video (Yellow)', processed_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


# Часть 2: Обработка видеопотока с камеры (серый объект)
def process_camera_feed():
    # Открытие видеопотока с камеры
    cap = cv2.VideoCapture(0)  # 0 - индекс камеры (обычно это встроенная камера)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра (серый цвет)
        processed_frame = process_frame(frame, color='gray')

        # Отображение кадра
        cv2.imshow('Camera Feed (Gray)', processed_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


# Основная программа
if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1. Обработка видеофайла 'catball.mp4' (желтый объект)")
    print("2. Обработка видеопотока с камеры (серый объект)")
    choice = input("Введите номер режима (1 или 2): ")

    if choice == "1":
        process_video_file()
    elif choice == "2":
        process_camera_feed()
    else:
        print("Неверный выбор. Завершение программы.")