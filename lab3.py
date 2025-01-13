import cv2
import time

# Загрузка предобученных моделей для детекции лиц, глаз и улыбок
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Инициализация видеозахвата
cap = cv2.VideoCapture(0)

# Переменные для расчета FPS
prev_time = 0
fps = 0

while True:
    # Захват кадра с камеры
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра")
        break

    # Конвертация кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Обработка каждого обнаруженного лица
    for (x, y, w, h) in faces:
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Область интереса (ROI) для глаз и улыбки
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Детекция глаз (только в верхней половине лица)
        roi_eyes_gray = roi_gray[0:int(h/2), 0:w]  # Верхняя половина лица
        eyes = eye_cascade.detectMultiScale(roi_eyes_gray, scaleFactor=1.1, minNeighbors=5)

        # Фильтрация: глаза должны находиться в верхней половине лица
        for (ex, ey, ew, eh) in eyes:
            # Проверяем, что глаза находятся в верхней половине лица
            if ey + eh < int(h/2):  # Координаты глаз не выходят за пределы верхней половины
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Если глаз меньше двух, выводим сообщение
        if len(eyes) < 2:
            cv2.putText(frame, "open your eyes", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Детекция улыбки
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        if len(smiles) == 0:  # Если улыбка не обнаружена
            cv2.putText(frame, "smile", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Рисуем прямоугольники вокруг улыбки
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    # Расчет FPS
    current_time = time.time()
    time_difference = current_time - prev_time
    if time_difference > 0:  # Избегаем деления на ноль
        fps = 1 / time_difference
    prev_time = current_time

    # Вывод FPS на экран
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Face Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()