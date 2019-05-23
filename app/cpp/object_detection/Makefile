# CC=gcc
# CXX=g++
# RM=rm -f
# CPPFLAGS=-g $(shell root-config --cflags)
# LDFLAGS=-g $(shell root-config --ldflags)
# LDLIBS=$(shell root-config --libs)

# SRCS=tool.cc support.cc
# OBJS=$(subst .cc,.o,$(SRCS))

# all: tool

# tool: $(OBJS)
#     $(CXX) $(LDFLAGS) -o tool $(OBJS) $(LDLIBS) 

# tool.o: tool.cc support.hh

# support.o: support.hh support.cc

# clean:
#     $(RM) $(OBJS)

# distclean: clean
#     $(RM) tool

CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g -Wall -Wextra -std=c++0x
LDFLAGS=
LDLIBS=-ltensorflow -lyaml-cpp -ljsoncpp $(shell pkg-config --cflags --libs opencv)
TARGET=benchmark

SRCS=object_detection.cpp utils/config_tools.cpp benchmark.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: benchmark

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDLIBS) 

object_detection.o: object_detection.cpp

clean:
	$(RM) $(OBJS) $(TARGET)