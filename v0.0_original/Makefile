CC = /opt/ibm/xlC/13.1.3/bin/xlc

#FILE = test_SOA.c
#PATH=/opt/ibm/xlf/15.1.0/bin/:/opt/ibm/xlC/13.1.0/bin/:$PATH
FILE = test_cpu.c

#CFLAGS = -O3 -qarch=pwr8 -qtune=pwr8 -qhot=level=1 -qalias=noansi
all: main

main:${FILE}
	${CC} ${CFLAGS} ${FILE} -o $@
main.S:${FILE}
	$(CC) $(CFLAGS) -S ${FILE} -o main.S
run: main
	./main
clean:
	-rm main
