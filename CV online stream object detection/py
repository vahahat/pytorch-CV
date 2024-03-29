import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Загрузка предварительно обученной модели детекции
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # переключение модели в режим вывода

# Функция для преобразования изображения для модели
def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)

# Проверяем, успешно ли было открыто видео
if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break
        
        # Преобразование кадра для модели
        tensor_frame = transform_image(frame)

        # Прогнозирование
        predictions = model(tensor_frame)

        # Детекция объектов на кадре (рисуем рамки вокруг распознанных объектов)
        for element in range(len(predictions[0]['boxes'])):
            boxes = predictions[0]['boxes'][element].cpu().numpy()
            score = predictions[0]['scores'][element].cpu().numpy()
            # Отрисовка рамок для объектов с уверенностью выше определенного порога
            if score > 0.5:
                frame = cv2.rectangle(frame,
                                      (int(boxes[0]), int(boxes[1])),
                                      (int(boxes[2]), int(boxes[3])),
                                      (0, 255, 0), 2)
        
        # Отображение обработанного кадра
        cv2.imshow('Object Detection', frame)

        # Выход из цикла по нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
