CFLAGS=-pedantic -O3

all:	mitchell hhj numberlines seeding baromhowell

clean:
	rm mitchell mitchell.exe hhj hhj.exe numberlines numberlines.exe seeding seeding.exe baromhowell baromhowell.exe *.o


mitchell:	mitchell.o subr.o
	$(CC) $(CFLAGS) -o mitchell mitchell.o subr.o

hhj:	hhj.o subr.o
	$(CC) $(CFLAGS) -o hhj hhj.o subr.o

baromhowell:	baromhowell.o subr.o
	$(CC) $(CFLAGS) -o baromhowell baromhowell.o subr.o

seeding: seeding.o subr.o
	$(CC) $(CFLAGS) -o seeding seeding.o subr.o

numberlines:	numberlines.o schedule.o subr.o usertime.o
	$(CC) $(CFLAGS) -o numberlines numberlines.o schedule.o subr.o usertime.o
