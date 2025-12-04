OS = $(shell uname -s)

CC 			?= 	cc
CXX			?= 	c++
DESTDIR		?= 	/usr/local
FLAGS 		+= 	-std=c++11
ifdef DEBUG
FLAGS		+=	-ggdb -O0
else
FLAGS		+=	-O0
endif

LIBS 		= 	
CFLAGS		+=	-I. $(FLAGS) -c -MMD -Wall -Wextra -Wno-unused-parameter -Werror -pedantic
LDFLAGS		+=	$(FLAGS)

SOURCES		= 	test.cxx
OBJECTS		= 	$(SOURCES:.cxx=.o)
DEPENDENCIES=	$(SOURCES:.cxx=.d)
EXECUTABLE	=	argstest

.PHONY: all clean pages runtests uninstall install installman

all: $(EXECUTABLE)

-include $(DEPENDENCIES)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

uninstall:
	-rm $(DESTDIR)/include/args.hxx
	-rmdir $(DESTDIR)/include
	-rm $(DESTDIR)/share/man/man3/args_*.3.bz2
	-rmdir -p $(DESTDIR)/share/man/man3

install:
	mkdir -p $(DESTDIR)/include
	cp args.hxx $(DESTDIR)/include

installman: doc/man
	mkdir -p $(DESTDIR)/share/man/man3
	cp doc/man/man3/*.3.bz2 $(DESTDIR)/share/man/man3

clean:
	rm -rv $(EXECUTABLE) $(OBJECTS) $(DEPENDENCIES) doc

pages:
	-rm -r pages/*
	doxygen Doxyfile
	cp -rv doc/html/* pages/

doc/man: 
	doxygen Doxyfile
	bzip2 doc/man/man3/*.3

runtests: ${EXECUTABLE}
	./${EXECUTABLE}

%.o: %.cxx
	$(CXX) $< -o $@ $(CFLAGS)
