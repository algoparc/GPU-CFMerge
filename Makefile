NVCC=nvcc
ARCH=-arch=sm_75
FLAGS=-O3 -use_fast_math -lineinfo --ptxas-options=-v -Xptxas -v,-warn-lmem-usage,-warn-spills
INCLUDE=-I./thrust/ -I./thrust/dependencies/cub/

default: \
	sort_int_worst_15.out \
	sort_int_random_15.out \
	sort_int_worst_17.out \
	sort_int_random_17.out 

sort_int_worst_15.out: test/sort_int_worst.cu
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDE) -DMY_E=15 -DMY_B=512 test/sort_int_worst2.cu -o $@

sort_int_worst_17.out: test/sort_int_worst.cu
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDE) -DMY_E=17 -DMY_B=256 test/sort_int_worst2.cu -o $@

sort_int_random_15.out: test/sort_int_random.cu
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDE) -DMY_E=15 -DMY_B=512 test/sort_int_random.cu -o $@

sort_int_random_17.out: test/sort_int_random.cu
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDE) -DMY_E=17 -DMY_B=256 test/sort_int_random.cu -o $@

clean:
	rm -f *.out	