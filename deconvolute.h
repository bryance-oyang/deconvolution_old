#ifndef _DECONVOLUTE_H_
#define _DECONVOLUTE_H_

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "tiff_goodness.h"
#include "opencl_utils.h"
#include "emalloc.h"

/* helper functions */
void init_images(char *input_image_filename, char *psf_image_filename);
void chunk_image();
void copy_input_image_to_chunk(int x, int y, int c);
void unchunk_image();
void copy_chunk_to_output_image(int x, int y, int c);
void output(char *output_image_filename);
void allocate_opencl_buffers();
void deconvolute_chunk(int chunk_index);
void do_iteration(int i);
void cleanup();

#endif /* !_DECONVOLUTE_H_ */
