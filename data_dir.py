# преобразует фото в 128 - мерную кодировку для тренировки распознования лиц
# благодаря чемуб сможем увеличить список фото человека для дальнейшей работы с ним
# суть проекта: сравнение кодировки сохраненной в базе данных с новой полученной кодировкой

import os
import pickle
import sys
import face_recognition
import cv2




# функция принимающая имя человека
def train_model_by_img(name):

    if not os.path.exists("dataset"):                                                        # проверка на наличие директории
        print("[ERROR] there is no directory 'dataset'")
        sys.exit()                                                                           # завершение работы скприта с помощью метода exit()

    known_encodings = []                                                                     # список пополняющихся кодировок
    images = os.listdir("dataset")                                                           # с помощью метода os.listdir передадим путь к директории датасета

    print(images)                                                                            # список всех достпуных изображений

    # итерация которая проверяет на идентичность изображений, если они одинковые добавляет их в список
    for(i, image) in enumerate(images):                                                      # enumerate() - индексирует изображения
        print(f"[+] processing img {i + 1}/{len(images)}")                                   # выводим результат
        # print(image)

        face_img = face_recognition.load_image_file(f"dataset/{image}")                      # загрузка изображения
        face_enc = face_recognition.face_encodings(face_img)[0]                              # извлекаем кодировку изображения

        print(face_enc)

        # реализация проверки каждого поступающего изображения с предыдущим, если результат сопоставления True, тогда добавляем новую кодировку в список
        if len(known_encodings) == 0:                                                        # если список пуст, добавляем первую кодировку
            known_encodings.append(face_enc)                        #
        else:                                                                                # иначе проходимся по постоянно изменяющейся его длине
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])   # сравниваем результат кодировки с уже существующим, кол-во сравнений зависит от длины списка
                # print(result)

                if result[0]:                                                                # если резульат один и тот же добавляем кодировку в список
                    known_encodings.append(face_enc)
                    print("Same person!")
                    break                                                                    # выходим из цикла for
                else:
                    print("Another person!")
                    break

    print(known_encodings)                                                                   # результат полученного списка
    print(f"Length {len(known_encodings)}")                                                  # распечатаем длину списка = кол-во его кодировок

   # сохранение данных
    data = {
        "name": name,
        "encodings": known_encodings
    }
    # преобразуем данные в поток байтов и сохраним в picle файл
    with open(f"{name}_encodings.pickle", "wb") as file:                                     # контекстное меню with, 1 аргумент - имя файла , 2 аругмент - "wb" запись в двоичном формате
        file.write(pickle.dumps(data))                                                       # вызываем сохранение нашего словаря

    return f"[INFO] File {name}_encodings.pickle successfully created"                       # возвращаем наш результат

# получение скриншота в реальном времени
def take_screenshot_from_video():
    cap = cv2.VideoCapture("/Users/ihortresnystkyi/Desktop/2022-12-18 16.27.39.mp4")         # путь к видеофайлу
    count = 0

    if not os.path.exists("dataset_from_video"):                                             # создание директории
        os.mkdir("dataset_from_video")

    while True:                                                                              # создаем цикл
        ret, frame = cap.read()                                                              # read() - захватывает и декодирует изображение
        fps = cap.get(cv2.CAP_PROP_FPS)                                                      # автоматизирует получение скриншотов
        multiplier = fps * 3                                                                 # множитель скриншота, где число - количество миллисекунд между кадрами
        print(fps)

        if ret:                                                                              # проверка: если с кадром все ок и его значение = True, реализуем код для получения скриншотов
            frame_id = int(round(cap.get(1)))                                                # получение значения текущего фрейма
            print(frame_id)
            cv2.imshow("frame", frame)                                                       # трансляция видео, 1 аргумент - название видео, 2 аргумент - сам фрейм
            k = cv2.waitKey(20)                                                              # реализовует клавишу, которая контролирует скорость воспроизведения видео

            # условие для получения скриншотов
            if frame_id % multiplier == 0:                                                   # условие для паузы видео
                cv2.imwrite(f"dataset_from_video/{count}.jpg", frame)                        # сохранение скриншотов, 1 аргумент - путь, 2 аргумент - формат
                print(f"Take a screenshot {count}")                                          # индексация изображений
                count += 1                                                                   # увеличение значения счетчика
            # создание кнопок для манипуляции с видео
            if k == ord(" "):                                                                # ord(" ") - в качетсве аргумента передаем нужную кнопку
                cv2.imwrite(f"dataset_from_video/{count}_extra_scr.jpg", frame)              # сохранение скриншотов
                print(f"Take an extra screenshot {count}")                                   # выводим результат
                count += 1                                                                   # увеличиваем значение счетчика
            elif k == ord("q"):                                                              # условие, если была нажата кнопка q
                print("Q pressed, closing the app")                                          # результат - закрытие окна видеовоспроизведения
                break                                                                        # выходим из цикла

        else:
            print("[Error] Can't get the frame...")
            break

    cap.release()                                                                            # закрытие видео
    cv2.destroyAllWindows()                                                                  # закрывает все открытые окна


def main():
     print(train_model_by_img("Person_name"))
     take_screenshot_from_video()


if __name__ == '__main__':
    main()


