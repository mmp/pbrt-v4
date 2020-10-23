##-----------------------------------------------------------------------------
# This allows each user to define a set of build customized settings
# according to your own configuration without modifying the project files.
# Include this empty file to avoid CMake error
##-----------------------------------------------------------------------------

if(MSVC)
    # some specific settings for MSVC here
endif()
	
if(UNIX AND NOT APPLE)
    # some Linux settings here
endif()
