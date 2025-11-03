from shared.load_library import sys
from shared.load_library import os
from shared.load_library import datetime, date
from shared.load_library import filedialog
from shared.load_library import cv
from shared.constants import FORMAT_IMAGE_FILE
from shared.constants import OUTPUT_PATH
from models.document import Document

class FileHandler:
    
    def select_file_path():
        """Метод для выбора изображения в файловой системе"""
        img_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", FORMAT_IMAGE_FILE), 
            ("Все файлы", "*.*")]
        )

        if img_path is None or img_path == "":
            #print("Файл не выбран")
            sys.exit("Файл не выбран")

        return img_path

    def save_image(image, copy = False):
        """Метод для сохранения изображения"""
        today = date.today()
        folder_path = OUTPUT_PATH +  datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        img_path = folder_path + "/img_doc_" + datetime.now().strftime("%H.%M.%S") 
        if copy is True:
            img_path += "copy"
        img_path += ".png"
        print(img_path)
        return cv.imwrite(img_path, image)
    
    def save_info_text(document):
        """Метод для сохранения информации о количестве строк и симловах"""
        today = date.today()
        folder_path = OUTPUT_PATH +  datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        txt_path = folder_path + "/info_doc_" + datetime.now().strftime("%H.%M.%S") + ".txt"
        try:
            with open(txt_path, "x", encoding="utf-8") as file:
                file.write(f"Количество строк: {document.count_lines}\n")
                file.write(f"Количество символов: {document.count_characters}\n")
            file.close()
        except FileExistsError:
            print("Файл уже существует!")