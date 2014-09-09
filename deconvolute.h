#ifndef _DECONVOLUTE_H_
#define _DECONVOLUTE_H_

#include <stdlib.h>
#include <stdio.h>
#include "tiff_goodness.h"
#include "opencl_utils.h"
#include "emalloc.h"

#define OUT_FILENAME "deconvoluted_image.tif"
#define CHUNK_SIZE 256

int n_iterations;

float ***chunks;
int chunk_size;
int n_chunks_x, n_chunks_y;

int width, height; /* dimensions of image to be deconvoluted */
int psf_width, psf_height; /* dimensions of psf */
uint16_t *input_image; /* image to be deconvoluted in RGBRGBRGB format */
uint8_t *psf_image;
uint16_t *output_image; /* deconvoluted image in RGBRGBRGB format */
/* converted to float images, each channel (RGB) is an array,
   i.e. normalized_input_image[3][width * height]
 */
float *normalized_input_image[3];
float *normalized_psf_image[3];
float *normalized_output_image[3];

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
/* events to wait on (sync) */
cl_event copy_events[3][3]; /* 0: image_a, 1: original, 2: psf */
cl_event kernel_events[3];

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
