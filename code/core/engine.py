import numpy as np
from PIL import Image
from PIL import ImageFilter
from core.scene import Light, LightProb
from progressbar import progressbar
from seaborn import distplot
import matplotlib.pyplot as plt

class Engine:

    def __init__(self, width, height):
        self.__width = width
        self.__height = height



    def render(self, scene, camera, max_depth=2, loops=1000):


        print("Starting rendering for resolution ", self.__width, "x", self.__height, ".")

        print("Creating rays from camera.")

        computed_colors = np.zeros((self.__width*self.__height, 3))
        sample_numbers = np.zeros(self.__width*self.__height)

        to_update = np.arange(self.__width*self.__height)

        for loop in range(loops):
            if to_update.shape[0] == 0:
                break
            rays = camera.create_rays(self.__width, self.__height)[to_update] # Contains rays origins and directions
            rays_eta = np.ones((rays.shape[0],), dtype=np.float) # Supposing the camera is in the void

            print("To update: ", to_update.shape[0], "pixels.")

            if loop >= 20:
                computed_colors_temp = self.__render(rays, rays_eta, scene, get_map=False, max_depth=max_depth)
                # Compute normalized distance between average of measures at rank loop and loop-1
                with np.errstate(divide='ignore', invalid='ignore'):
                    distance = np.max(np.abs(loop/(loop-1)*np.divide(computed_colors[to_update], computed_colors[to_update]+computed_colors_temp) - 1), axis=1)
                distance[np.isnan(distance)] = 1

                computed_colors[to_update] += computed_colors_temp
                sample_numbers[to_update] += 1

                to_update = to_update[np.argwhere(distance > 0.0001).reshape(-1)]


            else:
                computed_colors_temp, normal_map, tested_boxs, edges_map = self.__render(rays, rays_eta, scene,
                                                                                         loop_number=0,
                                                                                         max_depth=max_depth)
                computed_colors += computed_colors_temp
                sample_numbers += 1


            self.__make_image(computed_colors/sample_numbers.reshape(-1, 1)).save("step.png")
            

        print("Construct image.")
        # Make images and some interesting maps
        plt.imshow(sample_numbers.reshape(self.__width, -1).T)
        plt.savefig("compute.pdf")

        computed_colors /= sample_numbers.reshape(-1, 1)
        image = self.__make_image(computed_colors)


        return image, tested_boxs, normal_map, edges_map







    def __render(self, rays, rays_eta, scene, loop_number=0, max_depth= 1, get_map=True):

        computed_colors = np.zeros((rays.shape[0], 3))

        # Trace first hit, get time list, hit object list, and number of visted boxs
        ts_list, objs_list, tested_boxs_list = scene.trace(rays)

        # Which rays have hit an object ?
        hit_rays = np.argwhere(ts_list < np.inf).reshape((-1,))
        hit_rays_eta = rays_eta[hit_rays]

        no_hit_rays = np.argwhere(ts_list == np.inf).reshape((-1,))

        # Points where we have an hit event
        hit_points = rays[hit_rays, 0] + rays[hit_rays, 1] * (
            ts_list[hit_rays].reshape((-1, 1)))  # Origins + t*Directions



        # Normals
        normals = np.empty((len(hit_rays), 3))
        for i in range(len(hit_rays)):
            normal = objs_list[hit_rays[i]].normal(hit_points[i])
            normals[i] = normal
        incidence_cosines = np.sum(normals * rays[hit_rays, 1], axis=1)

        # Make sure the normals are well oriented and incidence_cosines are positives
        to_reorient = np.where(incidence_cosines > 0)
        normals[to_reorient] *= -1
        incidence_cosines = np.abs(incidence_cosines)


        if loop_number < max_depth:
            objects_eta = np.array(tuple(objs_list[x].eta for x in hit_rays))
            # If object_eta is same than ray_eta, that means we were inside an object and we are going to get out
            # We can then suppose we are entering an object which is the void, so we will rewrite object_eta to 1.
            objects_eta[np.where(objects_eta == rays_eta[hit_rays])] = 1.

            objects_specular = np.array(tuple(objs_list[x].specular for x in hit_rays))

            inside_square_root = 1 - np.square(hit_rays_eta / objects_eta) * (1 - np.square(incidence_cosines))
            inside_square_root[np.where(inside_square_root<0)] = 0

            transmited_cosines = np.sqrt(inside_square_root)

            # Compute Fresnell reflection coefficient
            reflection = (np.square((hit_rays_eta * incidence_cosines - objects_eta * transmited_cosines) /
                                    (hit_rays_eta * incidence_cosines + objects_eta * transmited_cosines)) +
                          np.square((hit_rays_eta * transmited_cosines - objects_eta * incidence_cosines) /
                                    (hit_rays_eta * transmited_cosines + objects_eta * incidence_cosines))
                          ) / 2
            refraction = 1 - reflection


            # From here we can use the refraction and reflections coefficients to computes boths rays (exponential cost)
            # We can also choose at random, stochastically, to do reflection or refraction, for monte carlo path tracing:

            refraction = (refraction > np.random.random(refraction.shape[0])).astype(np.int)
            reflection = 1 - refraction



            reflected_index = np.where((reflection > 0.01) & (objects_specular > 0.01))
            refracted_index = np.where((refraction > 0.01) & (objects_specular > 0.01))

            reflected_direction = rays[hit_rays[reflected_index], 1] - \
                                  2 * np.sum(normals[reflected_index] * rays[hit_rays[reflected_index], 1],
                                             axis=1).reshape(
                -1, 1) * normals[reflected_index]
            reflected_direction /= np.linalg.norm(reflected_direction, axis=1).reshape(-1, 1)
            reflected_rays = np.array((hit_points[reflected_index], reflected_direction)).transpose(1, 0, 2)
            reflected_rays_eta = rays_eta[hit_rays[reflected_index]]

            refracted_direction = (hit_rays_eta[refracted_index] / objects_eta[refracted_index]).reshape(-1, 1) \
                                  * (rays[hit_rays[refracted_index], 1] + incidence_cosines[refracted_index].reshape(-1,
                                                                                                                     1) *
                                     normals[refracted_index]) \
                                  - transmited_cosines[refracted_index].reshape(-1, 1) * normals[refracted_index]
            refracted_rays = np.array((hit_points[refracted_index], refracted_direction)).transpose(1, 0, 2)
            refracted_rays_eta = objects_eta[refracted_index]

            reflected_colors = self.__render(reflected_rays, reflected_rays_eta, scene, loop_number=loop_number + 1,
                                             max_depth=max_depth)
            refracted_colors = self.__render(refracted_rays, refracted_rays_eta, scene, loop_number=loop_number + 1,
                                             max_depth=max_depth)

            objects_specular = objects_specular.reshape(-1, 1)
            computed_colors[hit_rays[reflected_index]] += objects_specular[reflected_index] \
                                                          * reflection[reflected_index].reshape(-1, 1) * reflected_colors
            computed_colors[hit_rays[refracted_index]] += objects_specular[refracted_index] \
                                                          * refraction[refracted_index].reshape(-1, 1) * refracted_colors






        # Lambertian
        if isinstance(scene.light, Light):
            if hit_points.shape[0] > 0:
                directions_to_light = scene.light.position() - hit_points
                distance_to_light = np.linalg.norm(directions_to_light, axis=1).reshape(-1, )
                directions_to_light /= distance_to_light.reshape(-1, 1)
                irradience_cosines = np.sum(normals * directions_to_light, axis=1)
                irradience_cosines = np.maximum(irradience_cosines, 0)
                rays_to_light = np.array((hit_points, directions_to_light)).transpose(1, 0, 2)
                to_light_ts_list, _, _ = scene.trace(rays_to_light)
                # todo : Lambertian decreasing power should be 2 for energy conservation
                lambertian_values = 0.0 + irradience_cosines
                #lambertian_values[np.where(to_light_ts_list < distance_to_light)] = 0

                objects_colors = np.array([obj.color() for obj in objs_list[hit_rays]])
                objects_diffuse = np.array(tuple(objs_list[x].diffuse for x in hit_rays))
                computed_colors[hit_rays] += (objects_colors * (objects_diffuse.reshape(-1, 1)) * (lambertian_values.reshape(-1, 1)))
        elif isinstance(scene.light, LightProb):
            objects_colors = np.array([obj.color() for obj in objs_list[hit_rays]])
            objects_diffuse = np.array(tuple(objs_list[x].diffuse for x in hit_rays))
            where_diffuse = np.argwhere(objects_diffuse>0.01).reshape(-1)
            if where_diffuse.shape[0] > 0:

                random_directions = np.random.normal(size=(where_diffuse.shape[0], 3))
                #random_directions = np.tile(np.random.normal(size=3), (where_diffuse.shape[0], 1))
                random_directions += normals[where_diffuse] * (3 - np.sum(normals[where_diffuse] * random_directions, axis=1)).reshape(-1, 1)
                random_directions /= np.linalg.norm(random_directions, axis=1).reshape(-1,1)
                rays_to_light = np.array((hit_points[where_diffuse], random_directions)).transpose(1, 0, 2)
                to_light_ts_list, _, _ = scene.trace(rays_to_light)
                # If ts is np.inf, no objects were intersected and the ray hit the lightprob at infinite
                hit_light_rays = np.argwhere(to_light_ts_list == np.inf).reshape(-1)
                light_colors = scene.light.getColorsFromDirections(random_directions[hit_light_rays])
                colors = objects_diffuse[where_diffuse[hit_light_rays]].reshape(-1, 1) \
                         * light_colors * objects_colors[where_diffuse[hit_light_rays]]
                computed_colors[hit_rays[where_diffuse[hit_light_rays]]] += colors

            # No hit:
            computed_colors[no_hit_rays] = scene.light.getColorsFromDirections(rays[no_hit_rays, 1])




        if loop_number == 0 and get_map:
            normal_map = self.__normal_map(normals, hit_rays)
            tested_boxs = self.__tested_boxs_map(tested_boxs_list)
            edges_map = self.__edges_map(objs_list)
            return computed_colors, normal_map, tested_boxs, edges_map

        return computed_colors





    def __tested_boxs_map(self, tested_boxs_list):
        tested_boxs = np.zeros((tested_boxs_list.shape[0], 3))
        tested_boxs[:] = tested_boxs_list[:, None]
        tested_boxs *= 254 / np.max(tested_boxs_list)
        tested_boxs = tested_boxs.astype(np.uint8).reshape(self.__width, -1, 3).transpose((1, 0, 2))
        tested_boxs = Image.fromarray(tested_boxs, mode="RGB")
        return tested_boxs

    def __edges_map(self, objs_list):
        obj_to_id = np.vectorize(lambda x: 0 if x is None else x.id())
        objs_map = obj_to_id(objs_list).reshape(self.__width, -1)
        edges_map = np.zeros_like(objs_map, dtype=np.bool)
        edges_map[:-1, :-1] += objs_map[:-1, :-1] != objs_map[1:, 1:]
        edges_map[:, :-1] += objs_map[:, :-1] != objs_map[:, 1:]
        edges_map[:-1, :] += objs_map[:-1, :] != objs_map[1:, :]
        edges_map = Image.fromarray(edges_map.astype(np.uint8).transpose(1, 0) * 255, mode="L")
        return edges_map

    def __make_image(self, computed_colors):


        soft_min = np.percentile(computed_colors, 5)
        computed_colors = np.maximum(computed_colors-soft_min, 0)
        soft_max = np.percentile(computed_colors, 95)
        computed_colors = np.minimum(computed_colors/soft_max, 1)



        gamma = 1
        computed_colors = np.power(computed_colors, 1/gamma) #gamma correction
        #distplot(computed_colors.reshape(-1), kde=False)
        #plt.show()
        computed_colors *= 255

        image = np.array(computed_colors, dtype=np.uint8)  # Unsigned int8 (from 0 to 255)
        image = image.reshape(self.__width, self.__height, 3)
        image = Image.fromarray(image.transpose((1, 0, 2)).astype(np.uint8), 'RGB')
        image = image.filter(ImageFilter.SMOOTH_MORE)
        return image

    def __normal_map(self, normals, hit_rays):
        normal_map = np.zeros((self.__width * self.__height, 3), dtype=np.uint8)
        normal_map[hit_rays] = ((normals + 1.) * 127).astype(np.uint8)
        normal_map = normal_map.reshape(self.__width, -1, 3).transpose((1, 0, 2))
        normal_map = Image.fromarray(normal_map, mode='RGB')
        return normal_map