percol_inc = [project_inc, common_inc, include_directories('include')]

eigen_dep = dependency('Eigen3', method: 'cmake',  modules: ['Eigen3::Eigen'], required: true).as_system()


main_deps = [
  dependency('nlohmann_json', required: true),
  lil_dep]

percol_lib = static_library('percol_lib', 
  ['src/quantum_cluster.cpp', 'src/monte_carlo.cpp'],
  include_directories: percol_inc,
  dependencies: [main_deps, eigen_dep]
  )

pyro_dilution_bin = executable('pyro_dilution', 'pyro_dilution.cpp',
  include_directories:percol_inc,
  dependencies: main_deps
  )

sq_pyrochlore_bin = executable('sq_pyrochlore', 'sq_pyrochlore.cpp',
  include_directories:percol_inc,
  link_with: [percol_lib],
  dependencies: [main_deps, eigen_dep, hdf5_dep]
 )


sq_pyrochlore_exact_bin = executable('sq_pyrochlore_exact', 'sq_pyrochlore_exact.cpp',
  include_directories:percol_inc,
  link_with: [percol_lib],
  dependencies: [main_deps, eigen_dep, hdf5_dep]
 )


dump_bin = executable('sq_pyrochlore_dump', 'sq_pyrochlore_dump.cpp',
  include_directories:percol_inc,
  link_with: [percol_lib],
  dependencies: [main_deps, eigen_dep, hdf5_dep]
 )
