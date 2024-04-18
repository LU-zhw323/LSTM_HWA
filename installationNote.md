```
$ ml anaconda3 mvapich2/2.3.4 hdf5/1.10.7
$ h5cc -showconfig
# I dont know why the HDF5 Version downbelow is 1.10.6 but I load 1.10.7
# Also the installation point is not the same as the path of hdf5/1.10.7
SUMMARY OF THE HDF5 CONFIGURATION
            =================================

General Information:
-------------------
                   HDF5 Version: 1.10.6
                  Configured on: Tue Nov 24 01:31:47 UTC 2020
                  Configured by: conda@b006a79a2a76
                    Host system: x86_64-conda-linux-gnu
              Uname information: Linux b006a79a2a76 4.15.0-1098-azure #109~16.04.1-Ubuntu SMP Wed Sep 30 18:53:14 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
                       Byte sex: little-endian
             Installation point: /share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw

Compiling Options:
------------------
                     Build Mode: production
              Debugging Symbols: no
                        Asserts: no
                      Profiling: no
             Optimization Level: high

Linking Options:
----------------
                      Libraries: static, shared
  Statically Linked Executables: 
                        LDFLAGS: -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,-rpath,/share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/lib -Wl,-rpath-link,/share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/lib -L/share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/lib
                     H5_LDFLAGS: 
                     AM_LDFLAGS:  -L/share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/lib
                Extra libraries: -lcrypto -lcurl -lrt -lpthread -lz -ldl -lm 
                       Archiver: /home/conda/feedstock_root/build_artifacts/hdf5_split_1606181309526/_build_env/bin/x86_64-conda-linux-gnu-ar
                       AR_FLAGS: cr
                         Ranlib: /home/conda/feedstock_root/build_artifacts/hdf5_split_1606181309526/_build_env/bin/x86_64-conda-linux-gnu-ranlib

Languages:
----------
                              C: yes
                     C Compiler: /home/conda/feedstock_root/build_artifacts/hdf5_split_1606181309526/_build_env/bin/x86_64-conda-linux-gnu-cc
                       CPPFLAGS: -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/include
                    H5_CPPFLAGS: -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L   -DNDEBUG -UH5_DEBUG_API
                    AM_CPPFLAGS:  -I/share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/include
                        C Flags: -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/include -fdebug-prefix-map=/home/conda/feedstock_root/build_artifacts/hdf5_split_1606181309526/work=/usr/local/src/conda/hdf5_split-1.10.6 -fdebug-prefix-map=/share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw=/usr/local/src/conda-prefix
                     H5 C Flags:  -std=c99  -pedantic -Wall -Wextra -Wbad-function-cast -Wc++-compat -Wcast-align -Wcast-qual -Wconversion -Wdeclaration-after-statement -Wdisabled-optimization -Wfloat-equal -Wformat=2 -Wno-format-nonliteral -Winit-self -Winvalid-pch -Wmissing-declarations -Wmissing-include-dirs -Wmissing-prototypes -Wnested-externs -Wold-style-definition -Wpacked -Wredundant-decls -Wshadow -Wstrict-prototypes -Wswitch-enum -Wswitch-default -Wundef -Wunused-macros -Wunsafe-loop-optimizations -Wwrite-strings -Wlogical-op -Wlarger-than=2560 -Wsync-nand -Wframe-larger-than=16384 -Wpacked-bitfield-compat -Wstrict-overflow=5 -Wjump-misses-init -Wunsuffixed-float-constants -Wdouble-promotion -Wtrampolines -Wstack-usage=8192 -Wvector-operation-performance -Wdate-time -Warray-bounds=2 -Wc99-c11-compat -Wnull-dereference -Wunused-const-variable -Wduplicated-cond -Whsa -Wnormalized -Walloc-zero -Walloca -Wduplicated-branches -Wformat-overflow=2 -Wformat-truncation=2 -Wimplicit-fallthrough=5 -Wrestrict -fstdarg-opt -s -Wno-inline -Wno-aggregate-return -Wno-missing-format-attribute -Wno-missing-noreturn -Wno-suggest-attribute=const -Wno-suggest-attribute=pure -Wno-suggest-attribute=noreturn -Wno-suggest-attribute=format -O3
                     AM C Flags: 
               Shared C Library: yes
               Static C Library: yes


                        Fortran: yes
               Fortran Compiler: /home/conda/feedstock_root/build_artifacts/hdf5_split_1606181309526/_build_env/bin/x86_64-conda-linux-gnu-gfortran
                  Fortran Flags: 
               H5 Fortran Flags:  -pedantic -Wall -Wextra -Wunderflow -Wimplicit-interface -Wsurprising -Wno-c-binding-type  -s -O2
               AM Fortran Flags: 
         Shared Fortran Library: yes
         Static Fortran Library: yes

                            C++: yes
                   C++ Compiler: /home/conda/feedstock_root/build_artifacts/hdf5_split_1606181309526/_build_env/bin/x86_64-conda-linux-gnu-c++
                      C++ Flags: -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw/include -fdebug-prefix-map=/home/conda/feedstock_root/build_artifacts/hdf5_split_1606181309526/work=/usr/local/src/conda/hdf5_split-1.10.6 -fdebug-prefix-map=/share/Apps/lusoft/opt/spack/linux-centos8-x86_64/gcc-8.3.1/anaconda3/2020.07-4obfocw=/usr/local/src/conda-prefix
                   H5 C++ Flags:   -pedantic -Wall -W -Wundef -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wconversion -Wredundant-decls -Winline -Wsign-promo -Woverloaded-virtual -Wold-style-cast -Weffc++ -Wreorder -Wnon-virtual-dtor -Wctor-dtor-privacy -Wabi -finline-functions -s -O
                   AM C++ Flags: 
             Shared C++ Library: yes
             Static C++ Library: yes

                           Java: no


Features:
---------
                   Parallel HDF5: no
Parallel Filtered Dataset Writes: no
              Large Parallel I/O: no
              High-level library: yes
                Build HDF5 Tests: yes
                Build HDF5 Tools: yes
                    Threadsafety: yes
             Default API mapping: v110
  With deprecated public symbols: yes
          I/O filters (external): deflate(zlib)
                             MPE: no
                      Direct VFD: no
              (Read-Only) S3 VFD: yes
            (Read-Only) HDFS VFD: no
                         dmalloc: no
  Packages w/ extra debug output: none
                     API tracing: no
            Using memory checker: yes
 Memory allocation sanity checks: no
          Function stack tracing: no
       Strict file format checks: no
    Optimization instrumentation: no
$ conda activate /share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit
$ export PATH="/share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit/bin:$PATH"
$ export CC=mpicc
$ export HDF5_MPI="ON"
# I found the path of hdf5/1.10.7 by: module show hdf5/1.10.7
$ export HDF5_DIR=/share/Apps/lusoft/opt/spack/linux-centos8-haswell/intel-2021.3.0/hdf5/1.10.7-pzt66bo
# This will automatically install both h5py and mpi4py
$ pip install --no-binary=h5py h5py



```