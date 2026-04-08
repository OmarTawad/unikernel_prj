#include <stdio.h>

int main(void)
{
	printf("UNISPLIT_RPI4_P2_MAIN_ENTRY\n");
	printf("UNISPLIT_RPI4_P2_MAIN_DONE\n");

	for (;;)
		__asm__ volatile("wfe");
}
