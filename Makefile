all:	server.exe

%.exe:	%.o
	nvcc -arch=sm_52 $< -o $@

%.o:	%.cpp
	nvcc -D_FORCE_INLINES -std=c++11 -O3 --use_fast_math -Wno-deprecated-gpu-targets -m64 -x cu -arch=sm_52 -lineinfo -Xptxas --warn-on-local-memory-usage -Xptxas --warn-on-spills -Xcompiler -fPIC -I. -dc $<

