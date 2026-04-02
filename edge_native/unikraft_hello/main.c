#include <stdio.h>

/* Import user configuration */
#ifdef __Unikraft__
#include <uk/config.h>
#endif

int main(int argc, char *argv[])
{
	int i;

	printf("Hello world from UniSplit T01 on QEMU ARM64!\n");
	printf("Arguments:");
	for (i = 0; i < argc; ++i)
		printf(" \"%s\"", argv[i]);
	printf("\n");

	return 0;
}
