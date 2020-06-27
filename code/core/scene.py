import numpy as np
np.seterr('raise')
from core.object import Group, Triangle, Sphere
from multiprocessing import Pool
from itertools import accumulate
import json
import re
from struct import *


class Light():
    def __init__(self, position):
        self.__position = np.array(position)

    def position(self):
        return self.__position


class LightProb():
    def __init__(self, uri):

        with open(uri, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
            type = f.readline().decode('latin-1')
            if "PF" in type:
                channels = 3
            elif "Pf" in type:
                channels = 1
            else:
                assert(False)
            # Line 2: width height
            line = f.readline().decode('latin-1')
            width, height = re.findall('\d+', line)
            width = int(width)
            height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
            line = f.readline().decode('latin-1')
            BigEndian = True
            if "-" in line:
                BigEndian = False

            # Slurp all binary data
            samples = width * height * channels
            buffer = f.read(samples * 4)

            # Unpack floats with appropriate endianness
            if BigEndian:
                fmt = ">"
            else:
                fmt = "<"
            fmt = fmt + str(samples) + "f"
            img = unpack(fmt, buffer)
            img = np.array(img).reshape(width, height, channels)
            img = img.transpose(1,0,2)
            assert (width==height)
        self.__img = img

    @property
    def img(self):
        return self.__img


    def getColorsFromDirections(self, directions):
        # formula from https://www.pauldebevec.com/Probes/
        directions = directions.T
        norms = np.sqrt(np.sum(directions ** 2, axis=0))
        norms[np.where(norms == 0)] = 1
        directions /= norms


        x_y_norms = np.sqrt(np.sum(directions[:2]**2, axis=0))
        x_y_norms[np.where(x_y_norms == 0)] = 1

        size2 = self.__img.shape[0]/2
        coordinates = (((directions[:2]*np.arccos(directions[2])*size2)/(np.pi*x_y_norms)) + size2).astype(np.int)
        colors = self.__img[tuple(coordinates)]

        return colors








class Scene():

    def __init__(self, light):

        self.__light = light
        self.__objects = list()
        self.__datastructure = self.__objects

    def addobject(self, object):
        self.__objects.append(object)


    def root(self):
        if len(self.__datastructure) > 1:
            self.__datastructure = [Group(self.__datastructure)]
        return self.__datastructure[0]



    @property
    def light(self):
        return self.__light

    def trace(self, rays, multiprocess=4):
        print("Tracing ", rays.shape[0], " rays with ", multiprocess, " threads.")
        ts_list = np.full((rays.shape[0]), np.inf)
        objs_list = np.full((rays.shape[0]), None, dtype=np.object)

        if multiprocess is None:
            ts_list, objs_list, tested_boxs_list = self.root().hit(rays, ts_list, objs_list)
        else:

            p = Pool(multiprocess)
            work = tuple(zip(*(np.array_split(rays, multiprocess),
                               np.array_split(ts_list, multiprocess),
                               np.array_split(objs_list, multiprocess))))
            results = tuple(zip(*p.starmap(self.root().hit, work)))
            ts_list = np.concatenate(results[0])
            objs_list = np.concatenate(results[1])
            tested_boxs_list = np.concatenate(results[2])
        return ts_list, objs_list, tested_boxs_list



    def elementary_objects(self):
        objects = list()
        for obj in self.__objects:
            objects.extend(obj.elementary_objects())
        return objects

    def __area(self, bounding_box):
        diff = bounding_box[1] - bounding_box[0]
        return 2*(diff[0]*(diff[1]+diff[2]) + (diff[1]*diff[2]))

    def __split(self, elementary_objects, axis=0):
        if len(elementary_objects) <= 1:
            return Group(elementary_objects)
        else:
            elements_boundaries = np.array([obj.bounding_box()[:, axis] for obj in elementary_objects]).T
            end_sort_index = np.argsort(elements_boundaries[1])

            min_boundary = np.min(elements_boundaries[0])
            max_boundary = elements_boundaries[1, end_sort_index[-1]]

            if max_boundary-min_boundary == 0:
                return Group(elementary_objects)

            boundings_box_splits = self.__bounding_boxes_splits(elementary_objects[end_sort_index])

            min_sah = len(elementary_objects)*self.__area(Group(elementary_objects).bounding_box())
            cut_at = 0

            for size_first_group in range(1, len(elementary_objects)):
                size_second_group = len(elementary_objects) - size_first_group


                first_group_bounding = boundings_box_splits[0][size_first_group]
                second_group_bounding = boundings_box_splits[1][size_second_group]

                first_area = self.__area(first_group_bounding)
                second_area = self.__area(second_group_bounding)
                sah = (first_area*size_first_group + second_area*size_second_group)
                if sah < min_sah:
                    min_sah = sah
                    cut_at = size_first_group
            if cut_at == 0:
                return Group(elementary_objects)


            group1 = self.__split(elementary_objects[end_sort_index[:cut_at]], axis=(axis+1)%3)
            group2 = self.__split(elementary_objects[end_sort_index[cut_at:]], axis=(axis+1)%3)

            return Group([group1, group2])

    def serialize(self, filepath):
        serial = self.root().serialize()
        with open(filepath, 'w') as outfile:
            json.dump(serial, outfile)
        return serial

    def unserialize(self, filepath):
        with open(filepath) as json_file:
            data = json.load(json_file)
        self.__datastructure = [self.__build_from_serial(data)]

    def __build_from_serial(self, data):
        if isinstance(data, list):
            return Group([self.__build_from_serial(subdata) for subdata in data])
        elif isinstance(data, dict):
            if data["type"] == "triangle":
                return Triangle(data["points"], data["normals"], color=[255,255,255])
            elif data["type"] == "sphere":
                return Sphere(data["center"], data["radius"], color=[255,255,255])
            else:
                print(data)
                return Exception
        else:
            print(data)
            raise Exception


    def __bounding_boxes_splits(self, ordered_objects):
        boundings = [obj.bounding_box() for obj in ordered_objects]
        first_group_boundings = tuple(accumulate( boundings, self.__sum_bounding_boxes))
        second_group_boundings = tuple(accumulate( boundings[::-1], self.__sum_bounding_boxes))
        return first_group_boundings, second_group_boundings


    def __sum_bounding_boxes(self, bounding1, bounding2):
        boundings = np.array((bounding1, bounding2))
        bounding = np.array((np.min(boundings[:,0,:], axis=0),  np.max(boundings[:,1,:], axis=0)))
        return bounding


    def optimize(self):
        elementary_objects = np.array(self.elementary_objects())
        group = self.__split(elementary_objects)
        assert(len(group.elementary_objects()) == len(elementary_objects))
        self.__datastructure = [group]
        print("Scene optimized with BVH and SAH.")








