CC = /opt/ibm/xlC/13.1.2/bin/xlc
#FILE = test_SOA.c
#PATH=/opt/ibm/xlf/15.1.2/bin/:/opt/ibm/xlC/13.1.2/bin:$PATH
FILE = t4.c
#CFLAGS = -O3 -g -qarch=pwr8 -qtune=pwr8 -qhot=level=2 -qsmp=noauto -qaltivec=be -qsimd -qlist -qdebug=npwr7ra

CFLAGS = -O3 -g -qarch=pwr8 -qtune=pwr8 -qhot=level=2 -qsmp=noauto -qaltivec=be -qsimd -qlist -qdebug=npwr7ra
all: main main_vsx

main:${FILE}
	${CC} ${CFLAGS} ${FILE} -o $@
main_vsx:${FILE}
	${CC} ${CFLAGS} ${FILE} -o $@ -DVSXOPT -DPREFETCHOPT
main.S:${FILE}
	$(CC) $(CFLAGS) -S ${FILE} -o main.S
run: main
	./main
clean:
	-rm main main_vsx
