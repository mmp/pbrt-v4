# CMakeRC - A Standalone CMake-Based C++ Resource Compiler

CMakeRC is a resource compiler provided in a single CMake script that can easily
be included in another project.

## What is a "Resource Compiler"?

For the purpose of this project, a _resource compiler_ is a tool that will
compile arbitrary data into a program. The program can then read this data from
without needing to store that data on disk external to the program.

Examples use cases:

- Storing a web page tree for serving over HTTP to clients. Compiling the web
  page into the executable means that the program is all that is required to run
  the HTTP server, without keeping the site files on disk separately.
- Storing embedded scripts and/or shaders that support the program, rather than
  writing them in the code as string literals.
- Storing images and graphics for GUIs.

These things are all about aiding in the ease of portability and distribution of
the program, as it is no longer required to ship a plethora of support files
with a binary to your users.

## What is Special About CMakeRC?

CMakeRC is implemented as a single CMake module, `CMakeRC.cmake`. No additional
libraries or headers are required.

This project was initially written as a "literate programming" experiment. [The process for the pre-2.0 version can be read about here](https://vector-of-bool.github.io/2017/01/21/cmrc.html).

2.0.0+ is slightly different from what was written in the post, but a lot of it
still applies.

## Installing

Installing CMakeRC is designed to be as simple as possible. The only thing
required is the `CMakeRC.cmake` script. You can copy it into your project
directory (recommended) or install it as a package and get all the features you
need.

For [vcpkg](https://github.com/microsoft/vcpkg) users there is a `cmakerc` [port](https://github.com/microsoft/vcpkg/tree/master/ports/cmakerc) that can be installed via `vcpkg install cmakerc` or by adding it to `dependencies` section of your `vcpkg.json` file.

## Usage

1. Once installed, simply import the `CMakeRC.cmake` script. If you placed the
   module in your project directory (recommended), simply use `include(CMakeRC)`
   to import the module. If you installed it as a package, use `find_package(CMakeRC)`.

2. Once included, create a new resource library using `cmrc_add_resource_library`,
   like this:

   ```cmake
   cmrc_add_resource_library(foo-resources ...)
   ```

   Where `...` is simply a list of files that you wish to compile into the
   resource library.

  You can use the `ALIAS` argument to immediately generate an alias target for
  the resource library (recommended):

  ```cmake
  cmrc_add_resource_library(foo-resources ALIAS foo::rc ...)
  ```

  **Note:** If the name of the library target is not a valid C++ `namespace`
  identifier, you will need to provide the `NAMESPACE` argument. Otherwise, the
  name of the library will be used as the resource library's namespace.

  ```cmake
  cmrc_add_resource_library(foo-resources ALIAS foo::rc NAMESPACE foo  ...)
  ```

3. To use the resource library, link the resource library target into a binary
   using `target_link_libraries()`:

   ```cmake
   add_executable(my-program main.cpp)
   target_link_libraries(my-program PRIVATE foo::rc)
   ```

4. Inside of the source files, any time you wish to use the library, include the
   `cmrc/cmrc.hpp` header, which will automatically become available to any
   target that links to a generated resource library target, as `my-program`
   does above:

   ```c++
   #include <cmrc/cmrc.hpp>

   int main() {
       // ...
   }
   ```

5. At global scope within the `.cpp` file, place the `CMRC_DECLARE(<my-lib-ns>)` macro
   using the namespace that was designated with `cmrc_add_resource_library` (or
   the library name if no namespace was specified):

   ```c++
   #include <cmrc/cmrc.hpp>

   CMRC_DECLARE(foo);

   int main() {
       // ...
   }
   ```

6. Obtain a handle to the embedded resource filesystem by calling the
   `get_filesystem()` function in the generated namespace. It will be
   generated at `cmrc::<my-lib-ns>::get_filesystem()`.

   ```c++
   int main() {
       auto fs = cmrc::foo::get_filesystem();
   }
   ```

   (This function was declared by the `CMRC_DECLARE()` macro from the previous
   step.)

   You're now ready to work with the files in your resource library!
   See the section on `cmrc::embedded_filesystem`.

## The `cmrc::embedded_filesystem` API

All resource libraries have their own `cmrc::embedded_filesystem` that can be
accessed with the `get_filesystem()` function declared by `CMRC_DECLARE()`.

This class is trivially copyable and destructible, and acts as a handle to the
statically allocated resource library data.

### Methods on `cmrc::embedded_filesystem`

- `open(const std::string& path) -> cmrc::file` - Opens and returns a
  non-directory `file` object at `path`, or throws `std::system_error()` on
  error.
- `is_file(const std::string& path) -> bool` - Returns `true` if the given
  `path` names a regular file, `false` otherwise.
- `is_directory(const std::string& path) -> bool` - Returns `true` if the given
  `path` names a directory. `false` otherwise.
- `exists(const std::string& path) -> bool` returns `true` if the given path
  names an existing file or directory, `false` otherwise.
- `iterate_directory(const std::string& path) -> cmrc::directory_iterator`
  returns a directory iterator for iterating the contents of a directory. Throws
  if the given `path` does not identify a directory.

## Members of `cmrc::file`

- `typename iterator` and `typename const_iterator` - Just `const char*`.
- `begin()/cbegin() -> iterator` - Return an iterator to the beginning of the
  resource.
- `end()/cend() -> iterator` - Return an iterator past the end of the resource.
- `file()` - Default constructor, refers to no resource.

## Members of `cmrc::directory_iterator`

- `typename value_type` - `cmrc::directory_entry`
- `iterator_category` - `std::input_iterator_tag`
- `directory_iterator()` - Default construct.
- `begin() -> directory_iterator` - Returns `*this`.
- `end() -> directory_iterator` - Returns a past-the-end iterator corresponding
  to this iterator.
- `operator*() -> value_type` - Returns the `directory_entry` for which the
  iterator corresponds.
- `operator==`, `operator!=`, and `operator++` - Implement iterator semantics.

## Members of `cmrc::directory_entry`

- `filename() -> std::string` - The filename of the entry.
- `is_file() -> bool` - `true` if the entry is a file.
- `is_directory() -> bool` - `true` if the entry is a directory.

## Additional Options

After calling `cmrc_add_resource_library`, you can add additional resources to
the library using `cmrc_add_resources` with the name of the library and the
paths to any additional resources that you wish to compile in. This way you can
lazily add resources to the library as your configure script runs.

Both `cmrc_add_resource_library` and `cmrc_add_resources` take two additional
keyword parameters:

- `WHENCE` tells CMakeRC how to rewrite the filepaths to the resource files.
  The default value for `WHENCE` is the `CMAKE_CURRENT_SOURCE_DIR`, which is
  the source directory where `cmrc_add_resources` or `cmrc_add_resource_library`
  is called. For example, if you say `cmrc_add_resources(foo images/flower.jpg)`,
  the resource will be accessible via `cmrc::open("images/flower.jpg")`, but
  if you say `cmrc_add_resources(foo WHENCE images images/flower.jpg)`, then
  the resource will be accessible only using `cmrc::open("flower.jpg")`, because
  the `images` directory is used as the root where the resource will be compiled
  from.

  Because of the file transformation limitations, `WHENCE` is _required_ when
  adding resources which exist outside of the source directory, since CMakeRC
  will not be able to automatically rewrite the file paths.

- `PREFIX` tells CMakeRC to prepend a directory-style path to the resource
  filepath in the resulting binary. For example,
  `cmrc_add_resources(foo PREFIX resources images/flower.jpg)` will make the
  resource accessible using `cmrc::open("resources/images/flower.jpg")`. This is
  useful to prevent resource libraries from having conflicting filenames. The
  default `PREFIX` is to have no prefix.

The two options can be used together to rewrite the paths to your heart's
content:

```cmake
cmrc_add_resource_library(
    flower-images
    NAMESPACE flower
    WHENCE images
    PREFIX flowers
    images/rose.jpg
    images/tulip.jpg
    images/daisy.jpg
    images/sunflower.jpg
    )
```

```c++
int foo() {
    auto fs = cmrc::flower::get_filesystem();
    auto rose = fs.open("flowers/rose.jpg");
}
```
