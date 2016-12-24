program_NAME := orb
program_SAVEDIR := bin
program_C_SRCS := $(wildcard src/*.c)
program_CXX_SRCS := $(wildcard src/*.cpp)
program_C_OBJS := ${program_C_SRCS:.c=.o}
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_OBJS := $(program_C_OBJS) $(program_CXX_OBJS)
program_INCLUDE_DIRS :=
program_LIBRARY_DIRS :=
program_LIBRARIES :=

CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))
STDFLAGS = "c++11"

.PHONY: all clean distclean

all: $(program_NAME)

$(program_NAME): $(program_OBJS)
	    $(LINK.cc) $(program_OBJS) -std=$(STDFLAGS) -o $(program_SAVEDIR)/$(program_NAME) && \
			ln -s `pwd`/$(program_SAVEDIR)/$(program_NAME) ~/Code/bin/ && \
			echo $(LINK.cc) && echo $(STDFLAGS)

clean:
		@- $(RM) ~/Code/bin/$(program_NAME)
	    @- $(RM) $(program_SAVEDIR)/$(program_NAME)
		@- $(RM) $(program_OBJS)

distclean: clean
