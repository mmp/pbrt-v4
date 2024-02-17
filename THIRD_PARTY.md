
pbrt-v4 makes use of the following third-party libraries and data. Thanks
to all of the developers who have made these available!

* [double-conversion](https://github.com/google/double-conversion)
* [googletest](https://github.com/google/googletest)
* [lodepng](https://lodev.org/lodepng/)
* [OpenEXR](http:://www.openexr.com)
* [Ptex](http://ptex.us/)
* [rply](http://w3.impa.br/~diego/software/rply/)
* [skymodel](https://cgg.mff.cuni.cz/projects/SkylightModelling/)
* [stb](https://github.com/nothings/stb)
* [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)
* [zlib](https://zlib.net/)

Thanks also to Anders Langlands, who provided the Sensor implementation
used in the film model and Syoyo Fujita for the cyhair converter.

pbrt-v4 also includes spectral data from the following sources:

* Glass refractive index tables from https://refractiveindex.info, public
  domain CC0.
* Camera sensor measurement data from https://github.com/ampas/rawtoaces,
  Copyright © 2017 Academy of Motion Picture Arts and Sciences.

## Install using vcpkg

```bash
vcpkg install double-conversion
vcpkg install glfw3
vcpkg install glad
vcpkg install lodepng
vcpkg install ptex
vcpkg install stb
vcpkg install rply
vcpkg install libdeflate
vcpkg install qoi
vcpkg install utf8proc
vcpkg install gtest
vcpkg install openexr
vcpkg install 'openvdb[nanovdb]'

cmake -B build -S . "-DCMAKE_TOOLCHAIN_FILE=/home/lizz/dev/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build -j
```
