CC = /opt/ibm/xlC/13.1.3/bin/xlc

#FILE = test_SOA.c
#PATH=/opt/ibm/xlf/15.1.0/bin/:/opt/ibm/xlC/13.1.0/bin/:$PATH
FILE = test_cpu.c

CFLAGS = -O3 -qhot=level=2 -qalias=noansi -qaltivec=be -qsimd -qalias=noansi -qsmp=noauto 
#CFLAGS = -O3 -qarch=pwr8 -qtune=pwr8 -qhot=level=2 -qalias=noansi -qaltivec=be -qsimd -qlist -qsmp=noauto 
#CFLAGS = -O4 -qpdf1 -qarch=pwr8 -qtune=pwr8 -qhot=level=2 -qalias=noansi -qaltivec=be -qsimd -qlist -qsmp=noauto 
all: main

main:${FILE}
	${CC} ${CFLAGS} ${FILE} -o $@ 
main.S:${FILE}
	$(CC) $(CFLAGS) -S ${FILE} -o main.S
run: main
	./main
clean:
	-rm main
