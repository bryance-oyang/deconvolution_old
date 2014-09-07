#ifndef _DECONVOLUTE_H_
#define _DECONVOLUTE_H_

#include <stdlib.h>
#include <stdio.h>
#include "tiff_goodness.h"
#include "opencl_utils.h"
#include "emalloc.h"

#define OUT_FILENAME "deconvoluted_image.tif"

int width, height; /* dimensions of image to be deconvoluted */
int psf_width, psf_height; /* dimensions of psf */
uint16_t *input_image; /* image to be deconvoluted in RGBRGBRGB format */
uint8_t *psf_image;
uint16_t *output_image; /* deconvoluted image in RGBRGBRGB format */
/* converted to float images, each channel (RGB) is an array,
   i.e. normalized_input_image[3][width * height]
 */
float **normalized_input_image;
float **normalized_psf_image;
float **normalized_output_image;

/* opencl vars */
cl_context context;
cl_command_queue queue;
cl_program program;

/* helper functions */
void init_images(char *input_image_filename, char *psf_image_filename);
void output(char *output_image_filename);
void cleanup();

#endif /* !_DECONVOLUTE_H_ */
