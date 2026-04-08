#include <stdio.h>
#include <uk/print.h>

int main(void)
{
    uk_pr_err("PI_UEFI_POF_BOOT_START\n");
    printf("PI_UEFI_POF_BOOT_START\n");

    uk_pr_err("PI_UEFI_POF_UART_OK\n");
    printf("PI_UEFI_POF_UART_OK\n");

    uk_pr_err("PI_UEFI_POF_DONE\n");
    printf("PI_UEFI_POF_DONE\n");
    return 0;
}
