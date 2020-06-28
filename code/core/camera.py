
import numpy as np
from math import pi


class Camera:

    def __init__(self, position, h_orientation, v_orientation, fov=pi/4):
        # The fov is given here the horizontal angle of vue in radian
        position = np.array(position)
        v_orientation = np.array(v_orientation)
        h_orientation = np.array(h_orientation)


        v_orientation_norm = np.linalg.norm(v_orientation)
        assert(v_orientation_norm != 0)


        h_orientation_norm = np.linalg.norm(h_orientation)
        assert (h_orientation_norm != 0)
        assert(v_orientation.dot(h_orientation.T) == 0)

        self.__position = position
        self.__v_orientation = v_orientation/v_orientation_norm
        self.__h_orientation = h_orientation/h_orientation_norm

        self.__vue_orientation = np.cross(self.__v_orientation, self.__h_orientation)
        self.__vue_orientation /= np.linalg.norm(self.__vue_orientation)



        self.__fov = fov


    def position(self):
        return self.__position

    def vueorientation(self):
        return self.__vue_orientation

    def horizontalorientation(self):
        return self.__h_orientation

    def verticalorientation(self):
        return self.__v_orientation

    def fov(self):
        return self.__fov



class ThinLensCamera:

    def __init__(self, position, h_orientation, v_orientation, fov=pi/4, radius=1, focal_distance=6):
        # The fov is given here the horizontal angle of vue in radian
        position = np.array(position)
        v_orientation = np.array(v_orientation)
        h_orientation = np.array(h_orientation)


        v_orientation_norm = np.linalg.norm(v_orientation)
        assert(v_orientation_norm != 0)


        h_orientation_norm = np.linalg.norm(h_orientation)
        assert (h_orientation_norm != 0)
        assert(v_orientation.dot(h_orientation.T) == 0)

        self.__position = position
        self.__v_orientation = v_orientation/v_orientation_norm
        self.__h_orientation = h_orientation/h_orientation_norm

        self.__vue_orientation = np.cross(self.__v_orientation, self.__h_orientation)
        self.__vue_orientation /= np.linalg.norm(self.__vue_orientation)

        self.__radius = radius
        self.__focal_distance = focal_distance


        self.__fov = fov


    def position(self):
        return self.__position

    def vueorientation(self):
        return self.__vue_orientation

    def horizontalorientation(self):
        return self.__h_orientation

    def verticalorientation(self):
        return self.__v_orientation

    def fov(self):
        return self.__fov

    def create_rays(self, width, height):
        horizontal_angle = self.__fov
        vertical_angle = horizontal_angle * height / width

        x_s = self.horizontalorientation().reshape(-1,1)\
              *(np.sin(np.arange(width)/width - 0.5)*horizontal_angle)
        y_s = np.sin(np.arange(height) / height - 0.5) \
              * vertical_angle \
              * self.verticalorientation().reshape(-1, 1)

        x_y = np.add(x_s[:,:,None], y_s[:,None,:])
        rays_orientations = self.__focal_distance*(self.vueorientation()[:,None,None] + x_y).transpose(1,2,0).reshape(-1, 3)

        # compute random shift
        shift = np.random.normal(scale=self.__radius/3, size=width*height).reshape(-1,1)*(self.horizontalorientation().reshape(1,3)) \
                + np.random.normal(scale=self.__radius/3,size=width*height).reshape(-1,1)*(self.verticalorientation().reshape(1,3)) #99.7 % rule of normal law
        shift = shift.reshape(-1, 3)

        rays_orientations -= shift
        rays_orientations /= np.linalg.norm(rays_orientations, axis=1, keepdims=True)

        rays_origins = np.tile(self.position(), width*height).reshape(-1, 3) + shift

        rays = np.array((rays_origins, rays_orientations)).transpose(1, 0, 2)

        return rays
