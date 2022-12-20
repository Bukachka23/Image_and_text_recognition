import face_recognition
from PIL import Image, ImageDraw
import pickle
import cv2


# face_locations - возвращает список кортежей, где каждый кортеж - это найденное лицо
# функция
def face_rec():
    human_face_img = face_recognition.load_image_file("/sol.jpg")                            # загрузка изображения
    human_face_location = face_recognition.face_locations(human_face_img)                    # получение координат лица, если на изображение несколько человек

    humans_team_img = face_recognition.load_image_file("sols.jpeg")
    humans_team_location  = face_recognition.face_locations(humans_team_img)

    print(human_face_location)
    print(humans_team_location)
    print(f"Found {len(human_face_location)} face(s) in this image")
    print(f"Found {len(humans_team_location)} face(s) in this image")
    # создадим визуальную рамку для лиц
    pil_img1 = Image.fromarray(human_face_img)                                               # конвертируем изображение в PIL формат
    draw1 = ImageDraw.Draw(pil_img1)

    for(top, right, bottom, left) in human_face_location:                                    # цикл для перебора списка кортежей лиц
        draw1.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)      # 1 аргумент - коордианты сторон, 2 аргумент - желаемый цвет в rgb, 3 аргумент - толщина рамки

    del draw1                                                                                # после цикла удаляем обьект draw
    pil_img1.save("img/new_gal1.jpg")                                                        # сохранение изображения

    pil_img2 = Image.fromarray(humans_team_img)
    draw2 = ImageDraw.Draw(pil_img2)

    for(top, right, bottom, left) in humans_team_location:
        draw2.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw2
    pil_img2.save("img/new_justice_league.jpg")

# функция извлечения лица
def extracting_faces(img_path):                                                              # где аргумент указывает на путь к изображению
    count = 0                                                                                # создадим счетчик, чтобы на каждое фото добавлялась цифра для различия фото
    faces = face_recognition.load_image_file(img_path)                                       # загрузка изображения
    faces_locations = face_recognition.face_locations(faces)                                 # нахождение лиц на фото

    for face_location in faces_locations:                                                    # цикл для перебора списка кортежей лиц
        top, right, bottom, left = face_location                                             # извлечение точек координат

        face_img = faces[top:bottom, left:right]                                             # нахождение лица на фото по координатам
        pil_img = Image.fromarray(face_img)                                                  # создание pillow изображения
        pil_img.save(f"img/{count}_face_img.jpg")                                            # сохранение изображения
        count += 1

    return f"Found {count} face(s) in this photo"                                            # возвращает кол-во найденных лиц

# сравнение и нескольких лиц
def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]                                # вызываем face_encodings для кодировки лица, возвращает 128 - мерную кодировку лица
    print(img1_encodings)

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]                                # [0] - индекс отображает 1 найденное лицо

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)                # вызваем compare_faces для сравнения лиц
    print(result)
    # проверка на достоверность фото человеку
    if result[0]:
        print("Welcome")
    else:
        print("Sorry")


#обнаружение лица человека на видео
def detect_person_in_video():
    data = pickle.loads(open("Person_name_encodings.pickle", "rb").read())                   # считываем датасет в двоичном формате
    video = cv2.VideoCapture("37.mp4")                                                       # запуск вебкамеры

    while True:
        ret, image = video.read()                                                            # метод ret - захватывает, декодирует и возвращает кадр из видео

        locations = face_recognition.face_locations(image, model="cnn")                      # получение массива координат лиц людей на изображении
        encodings = face_recognition.face_encodings(image, locations)                        # получение координат лиц на изображении

        # реализация проверки каждого поступающего изображения с предыдущим, если результат сопоставления True, тогда выводим имя человека на изображении
        for face_encoding, face_location in zip(encodings, locations):
            result = face_recognition.compare_faces(data["encodings"], face_encoding)        # получение кодировок, 1 аргумент - из датасета, 2 аргумент - с изображения
            match = None

            # если результат True подставляем имя человека в переменную match
            if True in result:
               match = data["name"]
               print(f"Match found! {match}")
            else:
               print("Be careful")


            # отрисовка рамки координат на изображении
            left_top = (face_location[3], face_location[0])                                  # координаты рамки
            right_bottom = (face_location[1], face_location[2])                              # координаты рамки
            color = [0, 255, 0]                                                              # цвет рамки
            cv2.rectangle(image, left_top, right_bottom, color, 4)                           # рисовка рамки

            left_bottom = (face_location[3], face_location[2])
            right_bottom = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)               # отрисовка прямоугольника, cv2.FILLED - заполняет пространство цветом
            cv2.putText(
                image,
                match,
                (face_location[3] + 10, face_location[2] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                 1,
                (255, 255, 255),
                4
            )                                                                                # вывод текста на рамке

        cv2.imshow("detect_person_in_video is running", image)                               # вызов видео

        k = cv2.waitKey(20)
        if k == ord("q"):                                                                    # закртыие клавиши по нажатию кнопки
            print("Q pressed, closing the app")
            break

# главная функция, вызывающая остальные функциии
def main():
     face_rec()
     print(extracting_faces("soka_1.jpeg"))                                                  # извлечение лица
     compare_faces("rb_1.jpeg")                                                              # сравнение лиц
     detect_person_in_video()



if __name__ == '__main__':
    main()