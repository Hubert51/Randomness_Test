#include <stdio.h>

main(argc, argv) char **argv; {
	int size;
	int round;
	int NS, EW;
	int nrounds, skipround, movement;

	bound_check(argc, "argument count", 2, 2);
	size = atoi(argv[1]);
	bound_check(size, "group size", 4, 100); /* Some sanity */
	nrounds = size;
	skipround = size+1;
	if (size%2==0) {
		nrounds--;
		skipround=size/2+1;
	}
	printf("%d\n", nrounds);		/* number of rounds */
	printf("%d\n", size);		/* number of meetings/round */
	printf("2\n");			/* two groups in a Mitchell */
	printf("%d\n", size);		/* number of pairs NS */
	printf("%d\n", size);		/* number of pairs EW */
	for (round = 1; round <= nrounds; round++) {
		for (NS=1; NS <=size; NS++) {
			movement = -round;
			if (round >= skipround)
				movement--;
			EW = (NS+movement+size)%size+1;
			printf("%d %d ", NS, EW+size);
		}
		printf("\n");
	}
	return 0;
}
