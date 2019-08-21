import pyglet

class PygletViewer():
    def __init__(self, width, height):
        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.width = width
        self.height = height

    def update(self, pixel):
        self.window.clear()
        pyglet.image.ImageData(self.width, self.height, 'RGB', pixel.tobytes(), self.width * -3).blit(0,0)
    
    def close(self):
        self.window.close()