#include "quantize.h"
#include "tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    float *input = NULL;
    size_t count = 0;
    int8_t *out = NULL;
    float scale = 1.0f;
    FILE *fp;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_f32_bin> <output_i8_bin> <scale_txt>\\n", argv[0]);
        return 2;
    }

    if (load_f32_file_dynamic(argv[1], &input, &count) != 0) {
        fprintf(stderr, "Failed to read input file: %s\\n", argv[1]);
        return 1;
    }

    out = (int8_t *) malloc(count * sizeof(int8_t));
    if (!out) {
        free(input);
        return 1;
    }

    if (quantize_int8_symmetric(input, count, out, &scale) != 0) {
        free(input);
        free(out);
        return 1;
    }

    if (write_i8_file(argv[2], out, count) != 0) {
        free(input);
        free(out);
        return 1;
    }

    fp = fopen(argv[3], "w");
    if (!fp) {
        free(input);
        free(out);
        return 1;
    }
    fprintf(fp, "%.9g\n", scale);
    fclose(fp);

    free(input);
    free(out);
    return 0;
}
