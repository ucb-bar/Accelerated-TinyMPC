mkdir -p build-rvv
cd build-rvv
cmake .. -DUSE_RVV=ON -DUSE_TYPE=float32
make -j8
cd ..

mkdir -p build-rvv-handopt
cd build-rvv-handopt
cmake .. -DUSE_RVV=ON-DUSE_TYPE=float32 -DUSE_HANDOPT=ON
make -j8
cd ..  

mkdir -p build-cpu
cd build-cpu
cmake .. -DUSE_RVV=OFF -DUSE_CPU=ON -DUSE_TYPE=float32
make -j8
cd ..

mkdir -p build-eigen
cd build-eigen
cmake .. -DUSE_RVV=OFF -DUSE_EIGEN=ON -DUSE_CPU=ON -DUSE_TYPE=float32
make -j8
cd ..

mkdir -p build-gemmini
cd build-gemmini
cmake .. -DUSE_RVV=OFF -DUSE_CPU=ON -DUSE_GEMMINI=ON -DUSE_HANDOPT=ON -DUSE_MATVEC=OFF
make -j8
cd ..