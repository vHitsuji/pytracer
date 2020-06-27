from abc import ABC, abstractmethod
import numpy as np
np.warnings.filterwarnings('ignore')



class Object(ABC):

    def __init__(self, name):
        self.__name = name


    @abstractmethod
    def hit(self, rays, ts_list, objs_list):
        raise NotImplementedError("An object must have a hit method.")

    @abstractmethod
    def elementary_objects(self):
        raise NotImplementedError("An object must have a method that returns all elementary objects.")

    @abstractmethod
    def translate(self, vector):
        raise NotImplementedError("An object must have method for translation.")

    @abstractmethod
    def bounding_box(self):
        raise NotImplementedError("An object must return a bounding box.")

    def __str__(self):
        return self.__name

    @abstractmethod
    def serialize(self):
        raise NotImplementedError("An object must has a serialize method.")



class ElementaryObject(Object):
    def __init__(self, name, diffuse=1, specular=1, eta=1.33, color=None):
        # eta = 0 is equivalent to full reflection and no refraction
        super().__init__(name)
        self.__diffuse = diffuse
        self.__specular = specular
        self.__eta = eta

        if color is None:
            color = np.random.rand(3)
            color /= np.linalg.norm(color)
        else:
            color = np.array(color)/255
        self.__color = np.array(color)

    @abstractmethod
    def normal(self, point):
        raise NotImplementedError("An elementary object must have a method that give the normal to a point.")

    def elementary_objects(self):
        return [self]

    @property
    def eta(self):
        return self.__eta

    @property
    def diffuse(self):
        return self.__diffuse

    @property
    def specular(self):
        return self.__specular

    def color(self):
        return self.__color


class Group(Object):
    def __init__(self, objects_list, name="group"):
        super().__init__(name)
        self.__objects = objects_list
        bounding = np.empty((2, len(objects_list), 3))
        for i, obj in enumerate(self.__objects):
            min_bound, max_bound = obj.bounding_box()
            bounding[0,i] = min_bound
            bounding[1,i] = max_bound
        self.__bounding_box = np.array((np.min(bounding[0], axis=0), np.max(bounding[1], axis=0)))

    def serialize(self):
        return [obj.serialize() for obj in self.__objects]

    def bounding_box(self):
        return self.__bounding_box

    def objects(self):
        return self.__objects

    def translate(self, vector):
        vector = np.array(vector)
        for obj in self.__objects:
            obj.translate(vector)
        self.__bounding_box[0] += vector
        self.__bounding_box[1] += vector

    def elementary_objects(self):
        elementary_list = list()
        for obj in self.__objects:
            elementary_list.extend(obj.elementary_objects())
        return elementary_list

    def hit(self, rays, ts_list,  objs_list):
        # Fist check if the ray can hit the bounding box, then try to hit each objects in the group
        origins = rays[:,0,:]
        directions = rays[:,1,:]
        diff = self.__bounding_box - origins[:,None,:]

        with np.errstate(divide='ignore', invalid='ignore'):
            bounds = diff/directions[:,None,:]

        bounds = np.nan_to_num(bounds)

        bounds = np.sort(bounds, axis=1)
        bounds = np.array((np.max(bounds[:,0,:], axis=1), np.min(bounds[:,1,:], axis=1))).transpose(1,0)
        with np.errstate(over='ignore'):
            span = bounds[:,1] - bounds[:,0]

        hit_index = np.argwhere((span >= 0) * ((bounds[:,1] >= 0) + (bounds[:,0] >= 0))).reshape((-1,)) # Boolean calculous


        tested_boxs_list = np.ones((rays.shape[0]))

        if hit_index.shape[0]>0:
            for obj in self.__objects:
                ts, objs, tested_boxs = obj.hit(rays[hit_index], ts_list[hit_index], objs_list[hit_index])
                ts_list[hit_index] = ts
                objs_list[hit_index] = objs
                tested_boxs_list[hit_index] += tested_boxs


        return ts_list, objs_list, tested_boxs_list



    def check_integrity(self):
        bounding_boxes = [obj.bounding_box() for obj in self.__objects]
        result = True
        for bounding_box in bounding_boxes:
            if np.any(self.__bounding_box[0] > bounding_box[0]):
                return False
            if np.any(self.__bounding_box[1] < bounding_box[1]):
                return False

        for obj in self.__objects:
            if isinstance(obj, Group):
                if not obj.check_integrity():
                    return False
        return True


class Sphere(ElementaryObject):

    def __init__(self, center, radius, color=None, diffuse=1, specular= 0, eta=1.33):

        super().__init__("sphere", color=color, diffuse=diffuse, specular=specular, eta=eta)
        self.__center = np.array(center)
        self.__radius = radius
        self.__radius_square = radius*radius
        self.__hash = hash((tuple(self.__center), self.__radius))

    def translate(self, vector):
        self.__center += vector

    def hit(self, rays, ts_list, objs_list):
        # assume ray_orientation is unitary
        # Let a,b,c,  be the coefficient of the polynomial equation to solve
        # a.t^2 + b.t +c = °
        o_c = rays[:,0] - self.__center
        # a = 1
        b = 2*np.sum(rays[:,1]*o_c, axis=1)
        c = np.sum(o_c*o_c, axis=1) - self.__radius_square
        det = b*b - 4*c
        has_solution = np.argwhere(det >= 0).reshape(-1,)
        root_det = np.sqrt(det[has_solution])

        root1 = (-b[has_solution]-root_det)/2
        root2 = (-b[has_solution]+root_det)/2
        root1[np.where(root1 <= 1e-5)] = np.inf
        root2[np.where(root2 <= 1e-5)] = np.inf
        root = np.minimum(root1, root2)

        has_hit = np.argwhere(root < np.inf).reshape(-1,)
        sooner_hit = np.argwhere(ts_list[has_solution[has_hit]] > root[has_hit]).reshape(-1,)
        ts_list[has_solution[has_hit[sooner_hit]]] = root[has_hit[sooner_hit]]
        objs_list[has_solution[has_hit[sooner_hit]]] = self

        tested_boxs_list = np.full((rays.shape[0]), 0)
        return ts_list, objs_list, tested_boxs_list

    def serialize(self):
        serial = dict()
        serial["type"] = "sphere"
        serial["center"] = tuple(self.__center)
        serial["orientation"] = tuple(self.__orientation)
        return serial

    def normal(self, point):
        normal = point-self.__center
        normal /= np.linalg.norm(normal)
        return normal


    def bounding_box(self):
        return np.array((self.__center - self.__radius, self.__center + self.__radius))

    def __str__(self):
        return "sphere"

    def id(self):
        return self.__hash




class Mesh(Group):

    def __init__(self, uri, position, color=None, diffuse=1, specular=0, eta=1.33):
        lines = open(uri).read().splitlines()
        vertices = list()
        faces_points = list()
        normals = list()
        faces_normals = list()
        for line in lines:
            line = line.split()
            if len(line) != 0:
                if line[0] == 'v':
                    vertices.append((float(line[1]), float(line[2]), float(line[3])))
                elif line[0] == 'vn':
                    normals.append((float(line[1]), float(line[2]), float(line[3])))
                elif line[0] == 'f':
                    if len(normals) == 0:
                        p1 = line[1]
                        p2 = line[2]
                        p3 = line[3]
                    else:
                        p1, _, n1 = line[1].split("/")
                        p2, _, n2 = line[2].split("/")
                        p3, _, n3 = line[3].split("/")
                        faces_normals.append((int(n1), int(n2), int(n3)))
                    faces_points.append((int(p1), int(p2), int(p3)))
        vertices = np.array(vertices, dtype=np.float) + position
        faces_points = np.array(faces_points, dtype=np.int)
        normals = np.array(normals, dtype=np.float)
        faces_normals = np.array(faces_normals, dtype=np.int)

        triangles = list()
        if normals.shape[0] != 0:
            for i in range(faces_points.shape[0]):
                triangles.append(Triangle(vertices[faces_points[i]-1], normals=normals[faces_normals[i]-1], color=color, diffuse=diffuse, specular=specular, eta=eta))
        else:
            for i in range(faces_points.shape[0]):
                triangles.append(Triangle(vertices[faces_points[i]-1], color=color, diffuse=diffuse, specular=specular, eta=eta))


        super().__init__(triangles, name=uri.split("/")[-1])



class Triangle(ElementaryObject):

    def __init__(self,points, normals=None, color=None, diffuse=1, specular=0, eta=1.33):
        super().__init__("triangle", color=color, diffuse=diffuse, specular=specular, eta=eta)
        self.__points = np.array(points, dtype=np.float)
        self.__point1, self.__point2, self.__point3 = self.__points
        self.__normals = None
        if normals is not None:
            self.__normals = np.array(normals)
        normal = np.cross(self.__point2 - self.__point1, self.__point3 - self.__point1)
        with np.errstate(divide="warn", invalid="warn"):
            normal /= np.linalg.norm(normal)
        self.__normal = normal

        self.__bounding_box = np.array((np.min(points, axis=0), np.max(points, axis=0)))
        self.__hash = hash(tuple(tuple(point) for point in self.__points))


    def id(self):
        return self.__hash

    def serialize(self):
        serial = dict()
        serial["type"] = "triangle"
        serial["points"] = tuple(tuple(x) for x in self.__points)
        if self.__normals is not None:
            serial["normals"] = tuple(tuple(x) for x in self.__normals)
        else:
            serial["normals"] = None
        return serial

    def translate(self, vector):
        self.__points += vector
        self.__bounding_box += vector


    def hit(self, rays, ts_list, objs_list):

        #return np.ones(len(ts_list)), objs_list
        # By cramer's rule
        a = self.__point2 - self.__point1
        b = self.__point3 - self.__point1
        c = -rays[:,1]
        d = rays[:,0] - self.__point1

        a_s = np.tile(a, (rays.shape[0], 1))
        b_s = np.tile(b, (rays.shape[0], 1))

        dets = np.linalg.det(np.array((a_s, b_s, c,)).transpose((1, 2, 0)))
        solvable1 = np.argwhere(dets != 0).reshape(-1,)

        beta = np.linalg.det(np.array((d[solvable1], b_s[solvable1], c[solvable1],)).transpose((1, 2, 0)))/dets[solvable1]
        solvable2 = np.argwhere(np.logical_and(beta>=0, beta <=1)).reshape(-1,)
        solvable12 = solvable1[solvable2]

        gamma = np.linalg.det(np.array((a_s[solvable12], d[solvable12], c[solvable12],)).transpose((1, 2, 0)))/dets[solvable12]
        solvable3 = np.argwhere(np.logical_and(gamma >= 0, gamma <= 1)).reshape(-1, )
        solvable123 = solvable12[solvable3]

        alpha = 1 - (beta[solvable2[solvable3]]+gamma[solvable3])
        solvable4 = np.argwhere(np.logical_and(alpha >= 0, alpha <= 1)).reshape(-1, )
        solvable1234 = solvable123[solvable4]

        ts = np.linalg.det(np.array((a_s[solvable1234], b_s[solvable1234], d[solvable1234],)).transpose((1, 2, 0)))/dets[solvable1234]
        solvable5 = np.argwhere(ts > 1e-5).reshape(-1, )
        solvable12345 = solvable1234[solvable5]


        sooner_hit = (np.argwhere(ts[solvable5] < ts_list[solvable12345])).reshape(-1,)
        solvable123456 = solvable12345[sooner_hit]

        ts_list[solvable123456] = ts[solvable5[sooner_hit]]
        objs_list[solvable123456] = self
        tested_boxs_list = np.full((rays.shape[0]), 0)
        return ts_list, objs_list, tested_boxs_list

    def normal(self, point):
        if self.__normals is None:
            return self.__normal
        else:
            A = self.__points[1:,:] - self.__points[0]
            b = point - self.__points[0]
            beta, gamma = np.linalg.lstsq(A.T, b)[0]
            alpha = 1 - beta - gamma
            coeffs = np.array((alpha, beta, gamma))
            normal = self.__normals.T.dot(coeffs)
            return normal/np.linalg.norm(normal)

    def bounding_box(self):
        return self.__bounding_box

    def __str__(self):
        return "triangle"

