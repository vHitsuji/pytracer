

from core.engine import Engine
from core.camera import Camera
from core.object import Sphere, Mesh, Triangle, Group
from core.scene import Scene, Light
import numpy as np
from time import time
from datetime import datetime, timedelta

"""
Scenario:
Create a Scene that store objects, the scene must include the optimised data structure.
Create a camera.
Create a light source.
Add objects to the Scene.
Generate data structure for the Scene with optimize().

Create an engine object with that contains methods for ray tracings.
Give the scene to the engine that outputs a picture.

Show the picture.

"""



if __name__ == "__main__":

    engine = Engine(100, 100)
    camera = Camera([0.1,0,0], [1,0,0], [0,-1,0])
    light = Light([0,0,-1000])
    scene = Scene(light)


    np.random.seed(3)
    for i in range(50):
        x, y, z, r = np.random.randn(4)
        scene.addobject(Sphere([x*50, y*50, 100+z*10], max(0,5+r*5)))

    scene.addobject(Sphere([0,0,0], 500, anti=True))

    #scene.addobject(Triangle([-200,-200, 200], [-200,200, 200], [200,-200, 200]))
    scene.addobject(Mesh("teapot.obj", [0, -1, 6])) #, color=[166, 64, 185]))
    #scene.addobject(Mesh("bunny.obj", [0, -1.5, 5], color=[166, 60, 185]))
    #scene.addobject(Mesh("sponza.obj", [0, -1, 0], color=[100, 100, 100]))

    print("optimization started")
    scene.optimize()
    print("optimized")






    start_t = time()
    image, tested_boxs, distances, normal_map, edges_map = engine.render(scene, camera)
    print("Time for rendering: ", str(timedelta(seconds=int(time()-start_t))))


    tested_boxs.save("tested_boxes.png")
    distances.save("distances.png")
    image.save("raytracer.png")
    edges_map.save("edgesmap.png")
    image.show()

    normal_map.save("normal_map.png")

    exit(0)
