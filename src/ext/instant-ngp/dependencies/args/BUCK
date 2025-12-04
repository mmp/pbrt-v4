prebuilt_cxx_library(
  name = 'args', 
  header_namespace = '', 
  header_only = True, 
  exported_headers = [
    'args.hxx', 
  ], 
  visibility = [
    'PUBLIC', 
  ],
)

cxx_binary(
  name = 'test', 
  header_namespace = '', 
  headers = [
    'catch.hpp', 
  ], 
  srcs = [
    'test.cxx', 
  ], 
  deps = [
    '//:args', 
  ], 
)
