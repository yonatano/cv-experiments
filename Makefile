NAME := orb
SRCDIR := src
SAVEDIR := bin

STDFLAGS = "c++11"

.PHONY: all clean

all: build

build:
	clang++ $(SRCDIR)/*.cpp -std=$(STDFLAGS) -o $(SAVEDIR)/$(NAME) -g && \
		ln -s `pwd`/$(SAVEDIR)/$(NAME) ~/Code/bin/$(NAME)

clean:
		@- $(RM) ~/Code/bin/$(NAME)
	    @- $(RM) $(SAVEDIR)/$(NAME)

new: clean build
