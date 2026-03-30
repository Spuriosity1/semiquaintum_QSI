
compare_bin = executable('sq_pyrochlore_compare', 'sq_pyrochlore_compare.cpp',
  include_directories:percol_inc,
  link_with: [percol_lib],
  dependencies: [main_deps, eigen_dep, hdf5_dep]
 )

test_worm_bin = executable('test_worm', 'test_worm.cpp',
  include_directories:percol_inc,
  link_with: [percol_lib],
  dependencies: [main_deps, eigen_dep, hdf5_dep]
 )


test_monopole_worm_bin = executable('test_monopole_worm', 'test_monopole_worm.cpp',
  include_directories:percol_inc,
  link_with: [percol_lib],
  dependencies: [main_deps, eigen_dep, hdf5_dep]
 )

bench_autocorr_bin = executable('bench_autocorr', 'bench_autocorr.cpp',
  include_directories:percol_inc,
  link_with: [percol_lib],
  dependencies: [main_deps, eigen_dep, hdf5_dep]
 )
