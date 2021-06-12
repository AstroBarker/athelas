include ../../../Build/Makefile_Build

CFLAGS = -Wall

all: Driver

# Driver: Initialization.o 
# 	$(CC) $(CFLAGS) $(FLAGS) -o Driver Initialization.o
Driver: Initialization.o
	$(CC) $(CFLAGS) $(FLAGS) -o Driver_$(MACHINE) Initialization.o $(splode) $(Libraries)

clean: 
	rm -f *.o

clobber: clean
	rm -f Driver_$(MACHINE)
	rm -rf Driver_$(MACHINE).dSYM