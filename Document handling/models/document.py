import cv2 as cv

class Document:
    def __init__(self, image):
        self.original_image = image
        self.height, self.width = image.shape[:2]
        self.transformed_img = None
        self.selected_text_image = None
        self.count_lines = 0
        self.count_characters = 0
    
    def get_original_image(self):
        return self.original_image

    def get_dimensions(self):
        return self.width, self.height
    
    def set_transformed_img(self, image):
        self.transformed_img = image

    def get_transformed_img(self):
        return self.transformed_img
    
    def set_selected_text_img(self, image, count_lines, count_characters):
        self.selected_text_image = image
        self.count_lines = count_lines
        self.count_characters = count_characters

    def get_selected_text_img(self):
        return self.selected_text_image
    
    def get_info_text_img(self):
        return self.count_lines, self.count_characters