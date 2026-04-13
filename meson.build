project('dmnd_dil_2', ['cpp', 'c'],
  version: '2.0.0',
  meson_version : '>= 1.5.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)


#mpi_root = get_option('mpi_root')
#if mpi_root != ''
#  message('Using custom MPI root: ' + mpi_root)
#
#  mpi_inc = include_directories(join_paths(mpi_root, 'include'))
#  mpi_libdir = join_paths(mpi_root, 'lib/release')
#  mpi_dep = declare_dependency(
#    include_directories: mpi_inc,
#    link_args: [
#      '-L' +mpi_libdir, '-lmpi']
#    )
#else
#  mpi_dep = dependency('mpi', required: true, language: 'cpp')
#endif



hdf5_inc = get_option('hdf5_inc')
hdf5_lib = get_option('hdf5_lib')

if hdf5_inc != '' or hdf5_lib != ''
  message('Using custom HDF5 at ' + hdf5_inc + ', lib at ' + hdf5_lib)

  hdf5_inc = include_directories(hdf5_inc)

  hdf5_dep = declare_dependency(
    include_directories: hdf5_inc,
    link_args: ['-L' + hdf5_lib, '-lhdf5']
  )
else
  hdf5_dep = dependency('hdf5', language: 'c', required: true)
endif

project_inc = []

lil_dep = dependency('lattice_lib')

subdir('common')
subdir('percol')

subdir('test')
