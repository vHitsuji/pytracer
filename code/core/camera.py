
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


