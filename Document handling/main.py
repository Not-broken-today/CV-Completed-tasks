
from core.document_transformation import DocumentTransformation
from core.text_analyzer import TextAnalyzer
from utils.file_handler import FileHandler
from shared.constants import DEBUG_IMAGE
from shared.load_library import cv
from shared.load_library import sys

def main():
    """ Главный модуль """
    print("=== ОБРАБОТЧИК ДОКУМЕНТОВ ===")

    # Получаем путь к входному изображению
    img_path = FileHandler.select_file_path()

    # Обнаружение и трансформация документа
    transform = DocumentTransformation()
    result = transform.process_document(img_path)
    if result is None:
        sys.exit()
    # Сохранение документа
    FileHandler.save_image(image=result.get_transformed_img())

    # Анализ текста
    analyzer = TextAnalyzer()
    result = analyzer.process_document(result)
    if result is None:
        sys.exit()
    # Сохранение документа
    FileHandler.save_image(image=result.get_selected_text_img(), copy=True)
    FileHandler.save_info_text(result)

    if DEBUG_IMAGE is True:
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()