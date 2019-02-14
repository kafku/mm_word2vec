TARGET=word2vec word2vec_zh #mm_word2vec

CXX=g++
OPTI = -Ofast
WARN = -Wall
OMP = -fopenmp
DBG = #-g
INCLUDE=-I/usr/include
CXX_FLAGS=$(DBG) $(WARN) $(OPTI) $(OMP) -std=c++11 -march=native -funroll-loops -ftemplate-depth=1010
LD_FLAGS=-lpthread -lboost_program_options



.PHONY : all clean
all: $(TARGET)

word2vec: main.cc word2vec.h v.h model_generated.h
	$(CXX) $(CXX_FLAGS) -o ./$@ $< $(LD_FLAGS)


word2vec_zh: main_zh.cc word2vec.h v.h model_generated.h
	$(CXX) $(CXX_FLAGS) -o ./$@ $< $(LD_FLAGS)

clean:
	rm $(TARGET)
