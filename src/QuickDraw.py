import pygame
import numpy as np
from FNN import FNN
from PIL import Image, ImageEnhance
import ctypes

class QuickDraw():
    """
    A Pygame application for drawing and real-time prediction using a feedforward neural network (FNN).
    Users can draw on a canvas, and predictions for the drawing are displayed in real-time.
    """
    def __init__(self):
        """
        Initialize the QuickDraw application.
        Sets up the Pygame display, canvas, fonts, and prediction probabilities.
        """
        pygame.init()
        pygame.display.set_caption("Quick, Draw!")
        ctypes.windll.user32.SetProcessDPIAware()

        self.canvas = np.zeros((1024, 1024))
        self.display = pygame.display.set_mode((1524, 1024))
        self.display.fill((255, 255, 255))
        self.font = pygame.font.Font(None, 72)
        pygame.draw.rect(self.display, (0, 0, 0), (0, 0, 1024, 1024))
        pygame.display.flip()

        self.running = True
        self.prev_pos = None 

        self.drawings_probs = {"angel": 0.0,
                               "basketball": 0.0,
                               "car": 0.0,
                               "cat": 0.0,
                               "crab": 0.0,
                               "dolphin": 0.0,
                               "helicopter": 0.0,
                               "mushroom": 0.0,
                               "octopus": 0.0,
                               "skull": 0.0}

    def draw(self):
        """
        Handle drawing on the canvas based on mouse input.
        Updates the canvas array and draws lines using the Bresenham algorithm.
        """
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            if 0 <= x < 1024 and 0 <= y < 1024:
                if self.prev_pos:
                    self.bresenham_line(self.prev_pos[0], self.prev_pos[1], x, y)
                self.prev_pos = (x, y)
            elif 1094 <= x < 1454 and 884 <= y < 1004:
                self.canvas = np.zeros((1024, 1024))
                self.prev_pos = None
        else: 
            self.prev_pos = None

    def draw_circle(self, cx, cy, radius):
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    if 0 <= x < 1024 and 0 <= y < 1024:
                        self.canvas[x, y] = 255

    def bresenham_line(self, x1, y1, x2, y2):
        """
        Draw a line on the canvas using the Bresenham algorithm.

        Parameters:
        x1, y1 (int): Starting coordinates of the line.
        x2, y2 (int): Ending coordinates of the line.
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        err = dx - dy

        while True:
            self.draw_circle(x1, y1, 8)

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def get_predictions(self):
        """
        Preprocess the canvas and get predictions from the FNN model.

        Returns:
        np.ndarray: Softmax probabilities for each drawing category.
        """
        image = Image.fromarray(self.canvas.astype(np.uint8), mode='L')
        resized_image = image.resize((32, 32), Image.BICUBIC)
        enhancer = ImageEnhance.Contrast(resized_image)
        enhanced_image = enhancer.enhance(3.0)
        image_array = np.transpose(np.array(enhanced_image))
        image_vector = image_array.reshape(-1) / 255.0

        fnn = FNN()
        activations, _ = fnn.FeedForward(image_vector)
        
        return activations[-1]
    
    def update_probabilities(self):
        """
        Update the prediction probabilities based on the current canvas drawing.
        """
        predictions = self.get_predictions()
        for i, key in enumerate(self.drawings_probs.keys()):
            self.drawings_probs[key] = predictions[i]

    def update_display(self):
        """
        Update the Pygame display with the current canvas, predictions, and UI elements.
        """
        canvas_surface = pygame.surfarray.make_surface(self.canvas)
        scaled_surface = pygame.transform.scale(canvas_surface, (1024, 1024))
        self.display.blit(scaled_surface, (0, 0))

        pygame.draw.rect(self.display, (7, 22, 48), (1024, 0, 500, 1024))

        y_offset = 20
        text_surface = self.font.render(f"Predictions:", True, (255, 255, 255))
        self.display.blit(text_surface, (1040, y_offset))
        y_offset += 80
        sorted_probs = dict(sorted(self.drawings_probs.items(), key=lambda item: item[1], reverse=True))
        for key, prob in sorted_probs.items():
            text_surface = self.font.render(f"{key}: {prob*100:.1f}%", True, (255, 255, 255))
            self.display.blit(text_surface, (1040, y_offset))
            y_offset += 80

        pygame.draw.rect(self.display, (37, 52, 78), (1094, 884, 360, 120))
        text_surface = self.font.render("Clear Display", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(1094 + 360 // 2, 884 + 120 // 2))
        self.display.blit(text_surface, text_rect)

        pygame.display.flip()

    def run(self):
        """
        Main application loop for the QuickDraw program.
        Handles events, updates the display, and processes predictions.
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.draw()
            self.update_display()
            self.update_probabilities()

        pygame.quit()

if __name__ == "__main__":
    quick_draw = QuickDraw()
    quick_draw.run()
