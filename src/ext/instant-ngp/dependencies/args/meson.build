project('args.hxx', ['cpp'],
  version: '6.3.0',
  default_options: 'cpp_std=c++11',
  license: 'MIT'
)

# This is a header-only lib, all we need to do is include it
args_dep = declare_dependency(
  include_directories: include_directories('.')
).as_system('system')

install_headers('args.hxx')

# examples
if get_option('examples')
executable('gitlike', sources: 'examples/gitlike.cxx', dependencies: args_dep)
executable('completion', sources: 'examples/completion.cxx', dependencies: args_dep)
endif

# tests
if get_option('unittests')
test('argstest', executable('argstest',
  sources: 'test.cxx',
  dependencies: args_dep
))

test('argstest-multiple-inclusion', executable('argstest-multiple-inclusion',
  sources: [ 'test/multiple_inclusion_1.cxx', 'test/multiple_inclusion_2.cxx' ],
  dependencies: args_dep
))

test('argstest-windows-h', executable('argstest-windows-h',
  sources: 'test/windows_h.cxx',
  dependencies: args_dep
))
endif

