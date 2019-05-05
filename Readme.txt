We have used opencv and ViennaCL library.
So opencv will be needed in compilation.
For ViennaCL library , compiler will require the path of ViennaCl Files.


command to compile and run the executable for different files:-


#openmp.cpp will use all the processor of the cpu
# Filename : openmp.cpp  
		"g++ openmp.cpp -I/home/ankit/Desktop/me766/project/ViennaCL-1.7.1 -lopencv_imgcodecs -lopencv_core -fopenmp" and "time ./a.out"

#serial.cpp use only one processor of cpu
# Filename : serial.cpp
		"g++ serial.cpp -I/home/ankit/Desktop/me766/project/ViennaCL-1.7.1 -lopencv_imgcodecs -lopencv_core" and "time ./a.out"

#final.cu will use gpu
# Filename : final.cu
		"nvcc final.cu -I/home/ankit/Desktop/me766/project/ViennaCL-1.7.1 -lopencv_imgcodecs -lopencv_core" and "time ./a.out"

#mixed.cu will use gpu
# Filename : mixed.cu
		"nvcc mixed.cu -I/home/ankit/Desktop/me766/project/ViennaCL-1.7.1 -lopencv_imgcodecs -lopencv_core" and "time ./a.out"

we have also defined some cuda function in mixed.cu

The ViennaCl-Library path is "home/ankit/Desktop/me766/project/ViennaCL-1.7.1" for my pc, one needs to change that. 