#ifndef _DECONVOLUTE_H_
#define _DECONVOLUTE_H_

#include <stdlib.h>
#include <stdio.h>
#include "tiff_goodness.h"
#include "opencl_utils.h"
#include "emalloc.h"

#define OUT_FILENAME "deconvoluted_image.tif"
#define N_ITERATIONS 10

int width, height; /* dimensions of image to be deconvoluted */
int *dimensions; /* same thing, but suitable for copying to opencl */
int psf_width, psf_height; /* dimensions of psf */
int *psf_dimensions; /* same thing, but suitable for copying to opencl */
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
size_t *global_work_size;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel convolution_kernel[3];
cl_kernel deconvolution_kernel[3];
/* opencl memory to store images */
cl_mem k_image_a[3];
cl_mem k_image_b[3];
cl_mem k_original_image[3];
cl_mem k_psf_image[3];
cl_mem k_temp_image[3];
cl_mem k_dimensions[3];
cl_mem k_psf_dimensions[3];
/* events to wait on (sync) */
cl_event copy_events[3][5];
cl_event kernel_events[3][2];

/* helper functions */
void init_images(char *input_image_filename, char *psf_image_filename);
void output(char *output_image_filename);
void copy_images_to_opencl();
void do_iteration(int i);
void cleanup();

#endif /* !_DECONVOLUTE_H_ */
