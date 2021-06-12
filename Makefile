MACHINE = $(SPLODE_MACHINE)

OPT_LEVEL = DEBUG
FLAGS     = $(FLAGS_$(OPT_LEVEL))

include ./Build/Makefile_Build

splode: $(splode)

clean: rm -f *.o
