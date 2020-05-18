from core.engine import Engine
from core.camera import Camera
from core.object import Sphere, Mesh, Triangle, Group
from core.scene import Scene, Light
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
    scene_id = "bunny"
    engine = Engine(512, 512)
    #Scenes definition:

    if scene_id == "teapot":
        camera = Camera([0,3,-6], [1,0,0], [0,-1,-0.5])
        light = Light([10,-10,-10])
        scene = Scene(light)
        scene.addobject(Mesh("teapot.obj", [0, 0, 0], color=[255, 255, 255]))
        scene.addobject(Triangle(((-10., 0., 10.),( 0., 0.,  -10.), ( 10., 0., 10.)), color=[255, 255, 255]))
    elif scene_id == "bunny":
        camera = Camera([0, 5, -15], [1, 0, 0], [0, -1, -0.3])
        light = Light([10, -20, -10])
        scene = Scene(light)
        scene.addobject(Mesh("bunny.obj", [0, 0, 0], color=[255, 255, 255]))
        scene.addobject(Triangle(((-100., 0., 100.), (0., 0., -100.), (100., 0., 100.)), color=[255, 255, 255]))
    elif scene_id == "sponza":
        camera = Camera([8, 1.5, -1], [0, 0, 1], [0, -1, 0], fov=0.95)
        light = Light([0, 10, 0])
        scene = Scene(light)
        scene.addobject(Mesh("sponza.obj", [0, 0, 0], color=[255, 255, 255]))

    # Scene optimization with BVH
    start_t = time()
    scene.optimize()
    print("Time for optimizing: ", str(timedelta(seconds=int(time()-start_t))))

    # Rendering
    start_t = time()
    image, tested_boxs, distances, normal_map, edges_map = engine.render(scene, camera)
    print("Time for rendering: ", str(timedelta(seconds=int(time()-start_t))))


    # Images saving
    tested_boxs.save("tested_boxes.png")
    edges_map.save("edgesmap.png")
    distances.save("distances.png")
    image.save("raytracer.png")
    normal_map.save("normal_map.png")
    image.show()

    exit(0)
