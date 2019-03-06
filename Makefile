TARGET=word2vec #word2vec_zh #mm_word2vec

CXX=g++
OPTI = -Ofast
WARN = -Wall
OMP = -fopenmp
DBG = -g
INCLUDE=-I./eigen-git-mirror
CXX_FLAGS=$(DBG) $(WARN) $(OPTI) $(OMP) -std=c++11 -march=native -funroll-loops -ftemplate-depth=1010
CXX_FLAGS += -DEIGEN_DEFAULT_TO_ROW_MAJOR
CXX_FLAGS += -DEIGEN_USE_BLAS
LD_FLAGS=-lpthread -lboost_program_options -lopenblaso -lglog -lhdf5 -lhdf5_cpp

HEADERS=$(ls *.h)



.PHONY : all clean
all: $(TARGET)

word2vec: main.cc $(HEADERS)
	$(CXX) $(CXX_FLAGS) -o ./$@ $< $(INCLUDE) $(LD_FLAGS)


word2vec_zh: main_zh.cc word2vec.h v.h model_generated.h
	$(CXX) $(CXX_FLAGS) -o ./$@ $< $(INCLUDE) $(LD_FLAGS)

clean:
	rm $(TARGET)
