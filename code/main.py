

from core.engine import Engine
from core.camera import Camera, ThinLensCamera
import json
from core.object import Sphere, Mesh, Triangle, Group
from core.scene import Scene, Light, LightProb
import matplotlib.pyplot as plt
import numpy as np
from time import time
from progressbar import progressbar
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




    engine = Engine(500, 500)

    camera = Camera([0,0,0], [1,0,0], [0,-1,0], fov=1)
    #camera = ThinLensCamera([0,0,0], [1,0,0], [0,-1,0], radius=0.0, focal_distance=6, fov=1)


    #light = Light([10,0,10])
    #light = LightProb("hdrmap/grace_probe.pfm")
    light = LightProb("hdrmap/stpeters_probe.pfm")

    scene = Scene(light)

    #scene.addobject(Sphere([0,0,0], 1000, anti=True))

    #scene.addobject(Triangle(([-1000, -1000, 200], [1000,-1000, 200], [1000,1000, 200]), color=[255,0,0]))
    #scene.addobject(Triangle(([-1000, -1000, 200], [1000,1000, 200], [-1000, 1000, 200])))

    #scene.addobject(Triangle(([-10000, -10000, -1000], [10000,-10000, -1000], [10000,10000, -1000])))
    #scene.addobject(Triangle(([-10000, -10000, -1000], [10000,10000, -1000], [-10000, 10000, -1000])))

    np.random.seed(10)
    for i in range(30):
        x, y, z, r = np.random.randn(4)
        choice = np.random.choice(a=[False, True])
        scene.addobject(Sphere([x*50, y*50, 100+z*10], 5+r*5, specular=1, diffuse=0, eta=2.33))

    #scene.addobject(Triangle([-200,-200, 200], [-200,200, 200], [200,-200, 200]))
    scene.addobject(Mesh("teapot.obj", [-1, -1, 6], color=[255, 255, 255], diffuse=0, specular=1, eta=0.0001))
    scene.addobject(Mesh("teapot.obj", [2, -1, 9], color=[255, 255, 255], diffuse=1, specular=0, eta=0.0001))
    #scene.addobject(Sphere([2, 0, 8], 1, specular=1, diffuse=0, eta=0.001))

    #scene.addobject(Mesh("bunny.obj", [0, -1.5, 5], color=[100, 100, 100]))
    #scene.addobject(Mesh("sponza.obj", [0, -1, 0], color=[100, 100, 100]))

    print("optimization started")
    scene.optimize()
    print("optimized")

    start_t = time()
    image, tested_boxs, normal_map, edges_map = engine.render(scene, camera)
    print("Time for rendering: ", str(timedelta(seconds=int(time()-start_t))))


    tested_boxs.save("tested_boxes.png")
    image.save("raytracer.png")
    edges_map.save("edgesmap.png")
    image.show()

    normal_map.save("normal_map.png")

    exit(0)
