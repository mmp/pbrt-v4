# CMakeLists.txt for pbrt command-line utilities

# Find Qt5 package for the pbrtqtgui
find_package(Qt5 COMPONENTS Widgets Gui Core REQUIRED)

# Enable Automatic Moc, Uic, and Rcc
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# --- PBRT Main Executable ---
# Assuming 'pbrt.cpp' and other pbrt sources are handled in a parent CMakeLists.txt
# or would be added here if this file was solely responsible for all cmd tools.
# For now, this file focuses on pbrtqtgui and assumes other executables
# like 'pbrt' and 'imgtool' are defined elsewhere or would be added similarly.

# --- PBRT QT GUI Launcher ---
add_executable(pbrtqtgui pbrtqtgui.cpp)

target_link_libraries(pbrtqtgui PRIVATE 
    Qt5::Widgets 
    Qt5::Gui 
    Qt5::Core
)

# If pbrtqtgui needed to link against pbrt's core library (it doesn't currently, as it's a launcher):
# target_link_libraries(pbrtqtgui PRIVATE pbrt_libpbrt) # Assuming pbrt_libpbrt is the library name

# Optional: Installation
# If other command-line tools from this directory are installed, pbrtqtgui should be too.
# Example (assuming 'bin' is the destination for other pbrt executables):
# install(TARGETS pbrtqtgui DESTINATION bin)

# --- Other command line tools ---
# Example for imgtool, if it were defined here:
# add_executable(imgtool imgtool.cpp)
# target_link_libraries(imgtool PRIVATE pbrt_libpbrt) # If it uses pbrt's core
# install(TARGETS imgtool DESTINATION bin)

# Add other cmd tools (cyhair2pbrt, nanovdb2pbrt, plytool, etc.) here
# if this CMakeLists.txt becomes the primary one for the cmd/ directory.
# Each would look similar to:
# add_executable(toolname toolname.cpp)
# (Optional: target_link_libraries if they link against pbrt_libpbrt or other libs)
# (Optional: install command)
