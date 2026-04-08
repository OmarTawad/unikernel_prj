#include <stdio.h>
#include <stdint.h>

#define RPI4_PHASE1_PL011_BASE 0xFE201000UL
#define RPI4_PHASE1_UARTDR_OFFSET 0x00UL
#define RPI4_PHASE1_UARTFR_OFFSET 0x18UL
#define RPI4_PHASE1_FR_TXFF (1U << 5)

static inline uint16_t rpi4_phase1_uart_read(uint64_t offset)
{
	return *((volatile uint16_t *)(RPI4_PHASE1_PL011_BASE + offset));
}

static inline void rpi4_phase1_uart_write(uint64_t offset, uint16_t value)
{
	*((volatile uint16_t *)(RPI4_PHASE1_PL011_BASE + offset)) = value;
}

static void rpi4_phase1_uart_putc(char ch)
{
	while (rpi4_phase1_uart_read(RPI4_PHASE1_UARTFR_OFFSET) &
	       RPI4_PHASE1_FR_TXFF)
		;

	rpi4_phase1_uart_write(RPI4_PHASE1_UARTDR_OFFSET,
			       (uint16_t)(ch & 0xff));
}

static void rpi4_phase1_uart_puts(const char *s)
{
	while (*s) {
		if (*s == '\n')
			rpi4_phase1_uart_putc('\r');
		rpi4_phase1_uart_putc(*s++);
	}
}

__attribute__((noreturn)) void unisplit_pi4_phase1_bypass_entry(void)
{
	rpi4_phase1_uart_puts("UNISPLIT_RPI4_P1_EARLY_ENTRY\n");
	rpi4_phase1_uart_puts("UNISPLIT_RPI4_P1_BOOT_START\n");
	rpi4_phase1_uart_puts("UNISPLIT_RPI4_P1_UART_OK\n");
	rpi4_phase1_uart_puts("UNISPLIT_RPI4_P1_DONE\n");

	for (;;)
		__asm__ volatile("wfe");
}

int main(void)
{
	printf("UNISPLIT_RPI4_P1_BOOT_START\n");
	printf("UNISPLIT_RPI4_P1_UART_OK\n");
	printf("UNISPLIT_RPI4_P1_DONE\n");

	return 0;
}
