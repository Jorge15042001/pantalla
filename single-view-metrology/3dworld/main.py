import logging
import sys

import pygame

from play3d.models import Model, Grid,Sphere,Cube 
from pygame_utils import handle_camera_with_keys
from play3d.three_d import Device, Camera
from play3d.utils import capture_fps

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
def getPoint(plane,size=1,depth=0,origin=[0,0]):
    if plane == "xy":
        return [[origin[0],origin[1],depth,1], [origin[0]+size,origin[1],depth,1], [origin[0]+size,origin[1]+size,depth,1], [origin[0],origin[1]+size,depth,1]]
    elif plane=="yz" :
        return [[depth,origin[0],origin[1],1], [depth,origin[0]+size,origin[1],1], [depth,origin[0]+size,origin[1]+size,1], [depth,origin[0],origin[1]+size,1]]
    elif plane=="xz" :
        return [[origin[0],depth,origin[1],1], [origin[0]+size,depth,origin[1],1], [origin[0]+size,depth,origin[1]+size,1], [origin[0],depth,origin[1]+size,1]]
    raise Exception("invalid plane "+plane)
class Square(Model):
    def __init__(self,plane,size=1,depth=0,origin=[0,0],**kwargs):
        super(Square, self).__init__( 
            rasterize=True,
            data=getPoint(plane,size,depth,origin),
            faces = [ [1,2,3], [1,3,4] ], 
            **kwargs)
        self.color=kwargs["color"]

black, white = (20, 20, 20), (230, 230, 230)


Device.viewport(1024, 768)
pygame.init()
screen = pygame.display.set_mode(Device.get_resolution())

# just for simplicity - array access, we should avoid that
x, y, z = 0, 1, 2

# pygame sdl line is faster than default one
line_adapter = lambda p1, p2, color: pygame.draw.line(screen, color, (p1[x], p1[y]), (p2[x], p2[y]), 1)
put_pixel = lambda x, y, color: pygame.draw.circle(screen, color, (x, y), 1)

Device.set_renderer(put_pixel, line_renderer=line_adapter)

grid = Grid(color=(255,255,255), dimensions=(30, 30),position=(0,0,0))

# be aware of different scaling of .obj samples. Only vertices and faces supported!
face1 = Square(plane="xy",wireframe=True,color=(255,0,0))
face2 = Square(plane="xy",wireframe=True,color=(255,0,0),depth=1)
face3 = Square(plane="yz",wireframe=True,color=(0,255,0))
face4 = Square(plane="yz",wireframe=True,color=(0,255,0),depth=1)
face5 = Square(plane="xz",wireframe=True,color=(0,0,255))
face6 = Square(plane="yz",wireframe=True,color=(0,0,255),depth=1)
c=Cube(color=(200,0,200),rasterize=True,position=(10,0,10))
sph = Sphere(color=(200,200,100),position=(10,10,1))
camera = Camera.get_instance()
# move our camera up and back a bit, from origin
camera.move(y=4,x=4, z=3)


@capture_fps
def frame():
    if pygame.event.get(pygame.QUIT):
        sys.exit(0)

    screen.fill(black)
    handle_camera_with_keys()  # we can move our camera
    grid.draw()
    #  beetle.draw()
    face1.draw()
    face2.draw()
    face3.draw()
    face4.draw()
    face5.draw()
    face6.draw()
    sph.draw()
    c.draw()
    #  suzanne.draw()

    #  suzanne.rotate(0, 1, 0).draw()
    pygame.display.flip()


while True:

    frame()
