

project('dmnd_dil_2', ['cpp', 'c'],
  version: '2.0.0',
  meson_version : '>= 1.5.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)


project_inc = ['include']

lil_dep = dependency('latlib')

subdir('common')
subdir('percol')
