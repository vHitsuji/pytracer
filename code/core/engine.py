import numpy as np
from PIL import Image
from PIL import ImageFilter

class Engine:

    def __init__(self, width, height):
        self.__width = width
        self.__height = height



    def render(self, scene, camera):
        image = np.zeros((self.__width, self.__height, 3))  # Unsigned int8 (from 0 to 255)

        print("Starting rendering for resolution ", self.__width, "x", self.__height, ".")
        print("Creating rays from camera.")



        rays = self.__create_rays(camera)

        ts_list, \
        objs_list, \
        tested_boxs_list = scene.trace(rays)


        tested_boxs = np.zeros((tested_boxs_list.shape[0], 3))
        tested_boxs[:] = tested_boxs_list[:,None]
        tested_boxs *= 254/np.max(tested_boxs_list)
        tested_boxs = tested_boxs.astype(np.uint8).reshape(self.__width, -1, 3).transpose((1,0,2))
        tested_boxs = Image.fromarray(tested_boxs, mode="RGB")



        print("Check hit rays with objects.")
        hit_rays = np.argwhere(ts_list<np.inf).reshape((-1,))
        hit_points = rays[hit_rays,0] + rays[hit_rays,1]*(ts_list[hit_rays].reshape((-1, 1)))

        obj_to_id = np.vectorize(lambda x: 0 if x is None else x.id())
        objs_map = obj_to_id(objs_list).reshape(self.__width, -1)

        edges_map = np.zeros_like(objs_map, dtype=np.bool)
        edges_map[:-1, :-1] += objs_map[:-1, :-1] != objs_map[1:, 1:]
        edges_map[:, :-1] += objs_map[:, :-1] != objs_map[:, 1:]
        edges_map[:-1, :] += objs_map[:-1, :] != objs_map[1:, :]
        edges_map = Image.fromarray(edges_map.astype(np.uint8).transpose(1,0)*255, mode="L")



        distances = np.zeros((self.__width*self.__height, 3))
        max_d = np.max(ts_list[hit_rays])
        min_d = np.min(ts_list[hit_rays])
        distances[hit_rays, 0] = 255
        distances[hit_rays, 2] =  (ts_list[hit_rays] - min_d)/(max_d-min_d)*254
        distances = distances.astype(np.uint8).reshape((self.__width, -1, 3)).transpose((1,0,2))
        distances = Image.fromarray(distances, mode="RGB")

        print("Creating reflected rays to light.")

        directions_to_light = scene.light_position() - hit_points
        directions_to_light /= np.linalg.norm(directions_to_light, axis=1).reshape(-1, 1)


        normals = np.empty((len(hit_rays), 3))
        for i in range(len(hit_rays)):
            normal = objs_list[hit_rays[i]].normal(hit_points[i])
            normals[i] = normal



        normal_map = np.zeros((self.__width*self.__height, 3), dtype=np.uint8)
        normal_map[hit_rays] = ((normals+1.)*127).astype(np.uint8)
        normal_map = normal_map.reshape(self.__width,  -1, 3).transpose((1,0,2))
        normal_map = Image.fromarray(normal_map, mode='RGB')


        print("Creating real reflected ray with Descartes laws.")
        reflected = rays[hit_rays, 1] - \
                    2*np.sum(normals*rays[hit_rays, 1] ,axis=1).reshape(-1,1)*normals
        reflected /= np.linalg.norm(reflected, axis=1).reshape(-1, 1)

        print("Computes cosines between real reflected ray and ray to the light.")
        cosines = np.sum(normals*directions_to_light, axis=1)
        cosines = np.maximum(cosines, 0)

        light_reflection = 2 * cosines.reshape(-1,1)* normals - directions_to_light
        light_reflection /= np.linalg.norm(light_reflection, axis=1).reshape(-1, 1)
        h_specular = directions_to_light - rays[hit_rays, 1]
        h_specular /= np.linalg.norm(h_specular, axis=1).reshape(-1, 1)
        specular_cosines = np.maximum(np.sum(light_reflection * h_specular, axis=1), 0)


        print("Apply cosines to the colors of objects.")
        colors = np.array([obj.color() for obj in objs_list[hit_rays]])
        computed_colors = np.maximum(np.minimum(colors*(
                0.4 + 0.6*cosines.reshape(-1, 1) + 0.5*(specular_cosines.reshape(-1, 1)**5))
                                     , 255), 0)

        print("Construct image.")
        for i in range(computed_colors.shape[0]):
            x, y = divmod(hit_rays[i], self.__height)
            image[x, y] = computed_colors[i]


        image = Image.fromarray(
            image.transpose((1, 0, 2)).astype(np.uint8), 'RGB'
            )
        image = image.filter(ImageFilter.SMOOTH_MORE)

        return image, tested_boxs, distances, normal_map, edges_map



    def __create_rays(self, camera):
        horizontal_angle = camera.fov()
        vertical_angle = horizontal_angle * self.__height / self.__width

        x_s = camera.horizontalorientation().reshape(-1,1)\
              *(np.sin(np.arange(self.__width)/self.__width - 0.5)*horizontal_angle)
        y_s = np.sin(np.arange(self.__height) / self.__height - 0.5) \
              * vertical_angle \
              * camera.verticalorientation().reshape(-1, 1)

        x_y = np.add(x_s[:,:,None], y_s[:,None,:])
        rays_orientations = (camera.vueorientation()[:,None,None] + x_y).transpose(1,2,0).reshape(-1, 3)
        rays_orientations /= np.linalg.norm(rays_orientations, axis=1, keepdims=True)

        rays_origins = np.tile(camera.position(), self.__width*self.__height).reshape(-1, 3)

        rays = np.array((rays_origins, rays_orientations)).transpose(1, 0, 2)
        return rays

