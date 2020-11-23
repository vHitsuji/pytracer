![](/img/resized000.png)
![](/img/resized001.png)
![](/img/resized002.png)
![](/img/resized003.png)

Find the final report article at:

[https://github.com/vHitsuji/pytracer/blob/master/report/report.pdf](https://github.com/vHitsuji/pytracer/blob/master/report/report.pdf)

# Pytracer
  We made a basic ray-tracer engine in Python from scratch. The engine
  supports spheres and triangles and can also load meshes in the obj
  format. Since ray-tracers involve a lot of geometrical calculations,
  we had to implement some optimization schemes such as acceleration
  data-structure or tracing packs of rays with SIMD support.



Introduction
============

This project is part of a course taken at the University of Tokyo which
is an introduction to image synthesis. A ray-tracer is a kind of 3D
engine that traces and simulates light rays to compute colors. This
involves various scientific fields such as Euclidean geometry (for rays
intersections and reflections with objects), photometry (for color
computations), and computer sciences (for computational optimization).
Not all reflection rays can be computed and, however, some heuristics
exist to set the number of reflections for each ray of light. Since a
lot of rays traced from the light source are lost and do not hit the
camera, we use the Fermat's backward principle to compute rays from the
camera and check if the ray finally hits a light source. This allows us
to trace a fixed amount of rays defined by the resolution of the screen
we are projecting the image on. Some mandatory tasks were defined to
frame the project:

Triangle intersection

 We were provided three 3D objects defined as a set of triangles. It
    was natural to start by implementing triangle intersections. For
    each vertex of triangles, a normal vector is provided. From that, it
    was also mandatory to implement barycentric normal interpolations
    known as Phong interpolation.

Acceleration data-structure

 Ray-tracers involve a lot of geometrical computations. For each
    pixel of the final picture, the naive way is to try to intersect a
    ray with each object in the scene. A simple calculation can show
    that this is not computationally tractable and we had to implement a
    data-structure to navigate among the objects with the
    divide-and-conquer principle. For personal preferences, we decided
    to implement BVH with SAH heuristic.

Statistics

We had to provide statistics such as the number of rays traced per
    second or the number of object intersection tests for a pixel.


Optimization
============

The data structure 
------------------

We decided to implement the Bounding Volume Hierarchy algorithm known as
BVH to recursively split a scene into two sub-scenes. The Surface Area
Heuristic was used to select the best split from every candidate. Some
dynamic programming optimization was made to compute the candidates
faster, however, the construction of the data structure is not linear in
the number of objects. Every elementary object implements a bounding box
computation method that is called to get the smallest axis-aligned box
that is bounding the object. This smallest box is used by the BVH
algorithm to computes the bounding box of the objects group resulting in
the construction of a candidate. The resulting data-structures can be
serialized and saved to be loaded and reused later.
Figure [6](#fig:dsteapot){reference-type="ref" reference="fig:dsteapot"}
shows an example of a scene rendered with a resolution of
$1000\times1000$ pixels and a representation of the number of nodes
traversals in the data-structure optimized by BVH. For each pixel, we
computed the number of group nodes traversed and tested elementary
objects. The brighter the pixel is, the more intersection computations
have been made. This representation naturally shows the topology of the
optimized data-structure. Figure [8](#fig:dsbunny){reference-type="ref"
reference="fig:dsbunny"} shows the same result but for the bunny.

![An image rendered from the teapot 3D objects and a representation of
the number of nodes traversals in the corresponding optimized
data-structure.](report/img/teapot.png)

![An image rendered from the teapot 3D objects and a representation of
the number of nodes traversals in the corresponding optimized
data-structure.](report/img/tbteapot.png)

![An image rendered from the bunny 3D objects and a representation of
the number of nodes traversals in the corresponding optimized
data-structure.](report/img/bunny.png)

![An image rendered from the bunny 3D objects and a representation of
the number of nodes traversals in the corresponding optimized
data-structure.](report/img/tbbunny.png)

Packs of rays and SIMD 
----------------------

We did a massive *numpy* implementation that allows us to do
computations with compiled C code and support of SIMD. In the modern
processors, SIMD allows doing computations of 128-bits operands, which
represents four 32-bits operands computations at a time. For each tested
objects, the candidate rays are processed as a single ray. We observed a
substantial improvement in the computation duration.

Multithreading 
--------------

We used the *multiprocessing* Python library to distribute work among
every CPU's thread. For instance, with a four-core CPU, the image will
be divided into 4 and every unit will compute a piece. We faced a
problem with the memory allocation used by this optimization since the
scene was copied for each thread. Future work would be to use the BVH
data-structure to give the relevant part of the scene to each thread.

Cosmetic
========

Photometry 
----------

For each pixel, we computed the brightness as a sum of the ambient, the
diffuse and specular brightnesses. This is the so-called Phong's model.
The diffuse and specular brightnesses depend on the normals of the hit
objects and the angle of incident and reflected rays. Computing the
reflected rays is done by applying the Descartes's law of reflection
(the reflected ray is the opposite of the symmetry of the incident ray
by the normal of the hit object).
Figure [11](#fig:lightteapot) shows the contribution of each nature of
luminosities.

![From left to right we sequentially add ambient, diffuse and specular
lighting.](report/img/ambientteapot.png)
![From left to right we sequentially add ambient, diffuse and specular
lighting.](report/img/diffuseteapot.png)

![From left to right we sequentially add ambient, diffuse and specular
lighting.](report/img/specularteapot.png)

### Phong interpolation 

We implemented barycentric interpolations of normals that gives a smooth
aspect to surfaces composed of triangles. We computed normal maps to
show improvement. The
Figure [13](#fig:normalteapot){reference-type="ref"
reference="fig:normalteapot"} shows the improvement on the teapot
object. The Figure [15](#fig:normalsponza){reference-type="ref"
reference="fig:normalsponza"} show the same improvement for the Sponza
palace.

![Normal maps of the teapot with and without Phong
interpolation.](report/img/nnteapot.png){

![Normal maps of the teapot with and without Phong
interpolation.](report/img/nteapot.png)

![Normal maps of the Sponza palace with and without Phong
interpolation.](report/img/nnsponza.png)

![Normal maps of the Sponza palace with and without Phong
interpolation.](report/img/nsponza.png)

We planned to implement Phong tessellation, but the intersection test
was very difficult to make and we focused on other points.

