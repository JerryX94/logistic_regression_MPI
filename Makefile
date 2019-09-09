OBJECTS	 = lr.o
LIBS	 = -l msmpi -L "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64"
INCLUDES = -I "C:/Program Files (x86)/Microsoft SDKs/MPI/Include"
lr: $(OBJECTS)
	gcc -o lr.exe $(OBJECTS) $(LIBS)
lr.o: lr.c
	gcc -c lr.c $(INCLUDES)

.PHONY: clean
clean:
	-del lr.exe $(OBJECTS)
