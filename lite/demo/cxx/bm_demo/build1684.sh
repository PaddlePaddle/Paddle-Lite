rm -f lib/bmcompiler
rm -f lib/pcie
ln -s $(pwd)/lib/bmcompiler1684 $(pwd)/lib/bmcompiler
ln -s $(pwd)/lib/pcie1684 $(pwd)/lib/pcie
mkdir ./build
cd ./build
cmake ..
make
cd ..
rm -rf ./build
