from utils.file_handler import FileHandler
from models.document import Document
from shared.constants import KERNEL, MAX_DIMENSION, DEBUG_INFO, DEBUG_IMAGE
from shared.load_library import cv
from shared.load_library import np
from shared.load_library import sys

class TextAnalyzer:

    def process_document(self, document):
        try:
            if document is None:
                print("Ошибка при загрузки документа")
                return None
            
            # Получаем изображение из документа
            image = document.get_transformed_img()
            if image is None:
                print("Нет изображения документа")
                return None

            # Делаем изображение черно-белым
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            
            # Превращаем в чисто черно-белое
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            
            # Соединяем текст в линии (делаем строки целыми)
            horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
            connected_text = cv.morphologyEx(binary, cv.MORPH_CLOSE, horizontal_kernel)

            # Считаем белые пиксели по горизонтали
            horizontal_sum = np.sum(connected_text, axis=1)
            
            # Находим строки текста
            lines = self._find_text_lines(horizontal_sum)
            
            total_words = 0
            
            result_image = image.copy()
            
            # Обрабатываем каждую строку для поиска слов
            for i, (start, end) in enumerate(lines):
                # Рисуем зеленую рамку вокруг строки
                cv.rectangle(result_image, (0, start), (image.shape[1], end), (0, 255, 0), 2)
                
                # Подписываем номер строки
                cv.putText(result_image, f"Line {i+1}", (10, start + 15), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Находим слова в строке
                words_count = self._find_words_in_line(binary, start, end, result_image)
                total_words += words_count
            
            if DEBUG_INFO is True:
                print(f"Обработка завершена. Найдено {len(lines)} строк, {total_words} слов")
            
            # Отображаем промежуточные результаты только в debug режиме
            if DEBUG_IMAGE is True:
                cv.imshow("Binary", binary)
                cv.imshow("Connected Lines", connected_text)
                cv.imshow("Detected Lines", result_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            
            # Сохраняем результат обратно в документ
            document.set_selected_text_img(result_image, len(lines), total_words)
            return document
            
        except Exception as e:
            print(f"Ошибка при обработке: {str(e)}")
            return document
    
    def _find_text_lines(self, horizontal_sum):
        """Находит строки текста по горизонтальной проекции"""
        lines = []
        in_line = False
        line_start = 0
        
        if len(horizontal_sum) == 0:
            return lines
        
        # Определяем порог для текста
        non_zero_values = horizontal_sum[horizontal_sum > 0]
        if len(non_zero_values) == 0:
            return lines
            
        threshold = np.mean(non_zero_values) * 0.3
        
        for i, pixel_sum in enumerate(horizontal_sum):
            if pixel_sum > threshold and not in_line:
                # Начало строки
                in_line = True
                line_start = i
            elif pixel_sum <= threshold and in_line:
                # Конец строки
                in_line = False
                line_end = i
                
                # Добавляем строку, если она достаточно высокая
                if line_end - line_start >= 5:
                    lines.append((line_start, line_end))
        
        # Добавляем последнюю строку, если документ заканчивается текстом
        if in_line:
            line_end = len(horizontal_sum) - 1
            if line_end - line_start >= 5:
                lines.append((line_start, line_end))
        
        return lines
    
    def _find_words_in_line(self, binary_image, line_start, line_end, result_image):
        """Находит слова в строке"""
        words_count = 0
        
        # Проверяем валидность координат
        if line_start >= line_end or line_start < 0 or line_end > binary_image.shape[0]:
            return 0
            
        # Вырезаем область строки
        line_region = binary_image[line_start:line_end, :]
        
        # Считаем белые пиксели по вертикали
        vertical_sum = np.sum(line_region, axis=0)
        
        in_word = False
        word_start = 0
        
        # Порог для определения слов
        non_zero_vertical = vertical_sum[vertical_sum > 0]
        if len(non_zero_vertical) == 0:
            return 0
            
        word_threshold = np.mean(non_zero_vertical) * 0.5
        
        for j, pixel_sum in enumerate(vertical_sum):
            if pixel_sum > word_threshold and not in_word:
                # Начало слова
                in_word = True
                word_start = j
            elif pixel_sum <= word_threshold and in_word:
                # Конец слова
                in_word = False
                word_end = j
                
                # Если слово достаточно широкое
                if word_end - word_start > 3:
                    words_count += 1
                    # Рисуем синюю рамку вокруг слова
                    cv.rectangle(result_image, 
                                (word_start, line_start), 
                                (word_end, line_end), 
                                (255, 0, 0), 1)
        
        # Обрабатываем последнее слово, если строка заканчивается текстом
        if in_word:
            word_end = len(vertical_sum) - 1
            if word_end - word_start > 3:
                words_count += 1
                cv.rectangle(result_image, 
                            (word_start, line_start), 
                            (word_end, line_end), 
                            (255, 0, 0), 1)
        
        return words_count