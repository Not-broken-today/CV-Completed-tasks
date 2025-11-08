from utils.file_handler import FileHandler
from models.document import Document
from shared.constants import KERNEL, MAX_DIMENSION, DEBUG_INFO, DEBUG_IMAGE
from shared.load_library import cv
from shared.load_library import np
from shared.load_library import sys

class DocumentTransformation:

    def __init__(self, max_dimension=MAX_DIMENSION):
        self.max_dimension = max_dimension  # Максимальный размер для обработки
    
    def process_document(self, image_path):
        try:
            document = Document(cv.imread(image_path))
            if document.get_original_image() is None:
                return False
            elif DEBUG_INFO is True:
                print("Изображение " + image_path + " загружено!")
            
            # Определяем коэффициент масштабирования
            scale_factor = self._calculate_scale_factor(document.get_dimensions())
            if DEBUG_INFO is True:
                print("scale_factor = ", scale_factor)

            # Масштабируем изображение для быстрой обработки
            small_img = self._resize_image(document.get_original_image(), scale_factor)
            
            # Создание модели документа из уменьшенного изображения
            gray_image = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY)

            # Предобработка изображения
            processed_img = self._preprocess_image(gray_image)
            
            # Сегментация документа на уменьшенном изображении
            mask = self._segment_document(small_img)
            
            # Поиск углов документа
            corners = self._find_corners(mask)
            if corners is None or len(corners) < 4:
                print("Не найдено достаточно углов для преобразования")
                return None
            
            # Масштабируем углы обратно к исходному размеру
            original_corners = corners / scale_factor
            
            # Сортировка углов для правильного преобразования
            sorted_corners = self._sort_corners(original_corners)
            
            # Применение перспективного преобразования к исходному изображению
            document.set_transformed_img(self._apply_perspective_transform(document.original_image, sorted_corners))

            # Сохранение результата
            return document

        except Exception as e:
            print(f"Ошибка при обработке: {str(e)}")
            return None

    def _calculate_scale_factor(self, img_shape):
        """Вычисление коэффициента масштабирования"""
        height, width = img_shape
        max_size = max(height, width)
        
        if max_size <= self.max_dimension:
            return 1.0
        
        return self.max_dimension / max_size

    def _resize_image(self, image, scale_factor):
        """Масштабирование изображения"""
        if scale_factor == 1.0:
            return image
            
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        
        return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)

    def _preprocess_image(self, gray_image):
        """Предобработка изображения"""
        blurred = cv.GaussianBlur(gray_image, (3, 3), 0)
        return cv.morphologyEx(blurred, cv.MORPH_CLOSE, KERNEL, iterations=2)
    
    def _segment_document(self, image):
        """Сегментация документа с использованием GrabCut на уменьшенном изображении"""
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Уменьшаем отступы для маленького изображения
        margin = max(5, min(image.shape[0], image.shape[1]) // 20)
        rectangle = [margin, margin, image.shape[1]-margin, image.shape[0]-margin]
        
        bgd_model = np.zeros((1,65), np.float64)
        fgd_model = np.zeros((1,65), np.float64)
        
        # Уменьшаем количество итераций для скорости
        cv.grabCut(image, mask, rectangle, bgd_model, fgd_model, 3, cv.GC_INIT_WITH_RECT)
        
        # Создание бинарной маски
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        
        # Используем меньшие ядра для морфологических операций
        kernel = np.ones((3,3), np.uint8)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)
        
        return mask2
    
    def _find_corners(self, mask):
        """Поиск углов документа на уменьшенном изображении"""
        mask_visual = mask * 255
        
        # Находим контуры
        contours, _ = cv.findContours(mask_visual, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Берем самый большой контур
        largest_contour = max(contours, key=cv.contourArea)
        
        # Увеличиваем epsilon для упрощения контура на маленьком изображении
        epsilon = 0.03 * cv.arcLength(largest_contour, True)
        approx = cv.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) >= 4:
            corners = approx.reshape(-1, 2)
            # Убедимся, что углы находятся внутри изображения
            return self._validate_corners(corners, mask.shape)
        else:
            # Используем хорошие особенности с меньшим минимальным расстоянием
            corners = cv.goodFeaturesToTrack(mask_visual, 4, 0.01, 20)
            if corners is not None:
                corners = np.int32(corners).reshape(-1, 2)
                return self._validate_corners(corners, mask.shape)
            return None

    def _validate_corners(self, corners, img_shape):
        """Проверка, что углы находятся в пределах изображения"""
        height, width = img_shape[:2]
        valid_corners = []
        
        for corner in corners:
            x, y = corner
            if 0 <= x < width and 0 <= y < height:
                valid_corners.append(corner)
        
        return np.array(valid_corners) if len(valid_corners) >= 4 else None
    
    def _sort_corners(self, corners):
        """Сортировка углов в порядке: верхний-левый, верхний-правый, нижний-правый, нижний-левый"""
        if len(corners) != 4:
            return corners
            
        corners = corners.reshape(4, 2)
        sorted_corners = np.zeros((4, 2), dtype=np.float32)
        
        sum_coords = corners.sum(axis=1)
        sorted_corners[0] = corners[np.argmin(sum_coords)]  # верхний-левый
        sorted_corners[2] = corners[np.argmax(sum_coords)]  # нижний-правый
        
        # Оставшиеся точки
        remaining = np.delete(corners, [np.argmin(sum_coords), np.argmax(sum_coords)], axis=0)
        
        diff = np.diff(remaining, axis=1)
        sorted_corners[1] = remaining[np.argmin(diff)]  # верхний-правый
        sorted_corners[3] = remaining[np.argmax(diff)]  # нижний-левый
        
        return sorted_corners
    
    def _apply_perspective_transform(self, image, corners):
        """Применение перспективного преобразования для выравнивания документа"""
        if len(corners) != 4:
            return image
            
        # Определяем размеры выходного изображения
        width = max(
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[2] - corners[3])
        )
        height = max(
            np.linalg.norm(corners[0] - corners[3]),
            np.linalg.norm(corners[1] - corners[2])
        )
        
        # Целевые точки для преобразования
        dst_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])
        
        if DEBUG_IMAGE is True:
            # Создаем копию изображения для отладки
            debug_image = image.copy()
            
            # Отмечаем угловые точки на изображении
            for i, corner in enumerate(corners):
                x, y = corner.astype(int)
                # Рисуем круги разных цветов для разных углов
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # красный, зеленый, синий, голубой
                cv.circle(debug_image, (x, y), 20, colors[i], -1)  # закрашенные круги
                cv.putText(debug_image, str(i), (x + 15, y + 5), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
            
            # Рисуем контур, соединяющий точки
            for i in range(4):
                start_point = corners[i].astype(int)
                end_point = corners[(i + 1) % 4].astype(int)
                cv.line(debug_image, tuple(start_point), tuple(end_point), (255, 255, 255), 2)
            
            # Показываем размеры на изображении
            cv.putText(debug_image, f'Output: {int(width)}x{int(height)}', (10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.imshow("Image corners", debug_image)
            cv.imwrite("Image corners.png", debug_image)

        # Матрица перспективного преобразования
        M = cv.getPerspectiveTransform(corners, dst_points)
        
        # Применяем преобразование
        return cv.warpPerspective(image, M, (int(width), int(height)))