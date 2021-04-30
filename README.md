# Ray-tracing/Ray-marching Program
Written in C++ and using OpenFrameWorks package.

This program allows for renderings of 3D objects using either ray-tracing algorithms or ray-marching techniques.

![improvement16](https://user-images.githubusercontent.com/11671833/116645155-0a368d80-a92a-11eb-8d53-9516f9b1edbe.jpg)

Ray-tracing:
1.     Can add basic shapes such as spheres, cubes, planes, and pyramids to be rendered.
2.     Provides diffuse and blinn-phong shading.
3.     Provides accurate shadows and reflections.
4.     Provides normal mapping.
5.     Prototype implementation of atmospheric light scattering using Rayleigh algorithms.


Ray-marching:
1.     Utilizes signed distance functions to represent shapes.
2.     Provides the same basic shapes but with the addition of a torus.
3.     Provides infinite repetition of shapes without increasing memory footprint.
4.     Provides signed distance function algebra to create complex objects.
5.     Contains method to apply Perlin-noise generation on objects (in-progress).


![render](https://user-images.githubusercontent.com/11671833/116644872-5c2ae380-a929-11eb-91ac-3b25d96bb265.png)


![render5](https://user-images.githubusercontent.com/11671833/116644804-38679d80-a929-11eb-926b-2de19fbf57a0.jpg)
