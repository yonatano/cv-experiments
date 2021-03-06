NAME := orb
SRCDIR := src
SAVEDIR := bin

IMGMAGICKARGS=`Magick++-config --ldflags --libs`
IMGMAGICKFLAGS=`Magick++-config --cxxflags --cppflags`
STDFLAGS = "c++11"

.PHONY: all build clean new

all: new

build:
	clang++ $(IMGMAGICKFLAGS) $(SRCDIR)/*.cpp -std=$(STDFLAGS) -o $(SAVEDIR)/$(NAME) -g $(IMGMAGICKARGS) -larmadillo && \
		ln -s `pwd`/$(SAVEDIR)/$(NAME) ~/Code/bin/$(NAME)

clean:
		@- $(RM) ~/Code/bin/$(NAME)
	    @- $(RM) $(SAVEDIR)/$(NAME)

fast:
	clang++ $(IMGMAGICKFLAGS) $(SRCDIR)/*.cpp -Ofast -std=$(STDFLAGS) -o $(SAVEDIR)/$(NAME) -g $(IMGMAGICKARGS) && \
		ln -s `pwd`/$(SAVEDIR)/$(NAME) ~/Code/bin/$(NAME) 

new: clean build
