#include "deconvolute.h"

/* malloc all images and read in input image and psf
   also normalizes the psf with the variable total[3]
   by channel
 */
void init_images(char *input_image_filename, char *psf_image_filename)
{
	int i;

	printf("\nInitializing images...\n");
	fflush(stdout);

	input_image = read_tiff(input_image_filename, &width, &height);
	psf_image = read_tiff8(psf_image_filename, &psf_width, &psf_height);
	output_image = emalloc(3 * width * height *
			sizeof(*output_image));

	for (i = 0; i < 3; i++) {
		normalized_input_image[i] = emalloc(width * height *
				sizeof(*(normalized_input_image[i])));
		normalized_psf_image[i] = emalloc(psf_width * psf_height
				* sizeof(*(normalized_psf_image[i])));
		normalized_output_image[i] = emalloc(width * height *
				sizeof(*(normalized_output_image[i])));
	}

	/* also add a background to make sure pixels are not 0
	 * (black) or else we may encounter division by 0 */
	for (i = 0; i < 3 * width * height; i++) {
		normalized_input_image[i%3][i/3] =
			(float)input_image[i]/UINT16_MAX
			+ DIV_BY_ZERO_PREVENTION;
	}

	float total[] = {0, 0, 0};
	for (i = 0; i < 3 * psf_width * psf_height; i++) {
		normalized_psf_image[i%3][i/3] = (float)psf_image[i];
		total[i%3] += (float)psf_image[i];
	}
	for (i = 0; i < 3 * psf_width * psf_height; i++) {
		normalized_psf_image[i%3][i/3] /= total[i%3];
	}
}

void chunk_image()
{
	int c, x, y;
	int index;

	chunk_size = CHUNK_SIZE;

	n_chunks_x = (width - chunk_size)/(chunk_size/2) + 1;
	if ((width - chunk_size) % (chunk_size/2) != 0) {
		n_chunks_x += 1;
	}

	n_chunks_y = (height - chunk_size)/(chunk_size/2) + 1;
	if ((height - chunk_size) % (chunk_size/2) != 0) {
		n_chunks_y += 1;
	}

	printf("\nChunking image...\ntotal chunks: %d\nn_chunks_x: %d\nn_chunks_y: %d\nchunk_size: %d\n",
			n_chunks_x * n_chunks_y, n_chunks_x, n_chunks_y,
			chunk_size);
	fflush(stdout);

	chunks = emalloc(n_chunks_x * n_chunks_y * sizeof(*chunks));

	for (x = 0; x < n_chunks_x; x++) {
		for (y = 0; y < n_chunks_y; y++) {
			index = y * n_chunks_x + x;
			chunks[index] = emalloc(3 *
					sizeof(*chunks[index]));

			for (c = 0; c < 3; c++) {
				copy_input_image_to_chunk(x, y, c);
			}
		}
	}
}

void copy_input_image_to_chunk(int x, int y, int c)
{
	int i, j;
	int chunk_index;
	int image_i, image_j;
	int image_index;
	float *current;

	chunks[y * n_chunks_x + x][c] = emalloc(chunk_size * chunk_size
			* sizeof(*(chunks[y * n_chunks_x + x][c])));
	current = chunks[y * n_chunks_x + x][c];

	for (i = 0; i < chunk_size; i++) {
		for (j = 0; j < chunk_size; j++) {
			chunk_index = j * chunk_size + i;

			image_i = x*chunk_size/2 + i;
			image_j = y*chunk_size/2 + j;

			if ((image_i >= width) || (image_j >= height)) {
				current[chunk_index] =
					DIV_BY_ZERO_PREVENTION;
			} else {
				image_index = image_j * width + image_i;
				current[chunk_index] =
					normalized_input_image[c][image_index];
			}
		}
	}
}

void unchunk_image()
{
	int x, y, c;

	printf("\nUnchunking image...\n");
	fflush(stdout);

	for (x = 0; x < n_chunks_x; x++) {
		for (y = 0; y < n_chunks_y; y++) {
			for (c = 0; c < 3; c++) {
				copy_chunk_to_output_image(x, y, c);
			}
		}
	}
}

void copy_chunk_to_output_image(int x, int y, int c)
{
	int i, j;
	int chunk_index;
	int image_i, image_j;
	int image_index;
	int lbound_x, lbound_y, ubound_x, ubound_y;
	float *current;

	current = chunks[y * n_chunks_x + x][c];

	/* determine bounds for loop to make sure edge pixels get copied
	 * otherwise will only copy center of images */
	lbound_x = chunk_size/4;
	lbound_y = chunk_size/4;
	ubound_x = 3 * chunk_size/4;
	ubound_y = 3 * chunk_size/4;
	if (x == 0 && y == 0) { /* upper left */
		lbound_x = 0;
		lbound_y = 0;
	} else if (x == 0 && y != 0 && y != n_chunks_y - 1) { /* flush left */
		lbound_x = 0;
	} else if (x == 0 && y == n_chunks_y - 1) { /* lower left */
		lbound_x = 0;
		ubound_y = chunk_size;
	} else if (x != 0 && x != n_chunks_x - 1 &&
			y == n_chunks_y - 1) { /* flush bottom */
		ubound_y = chunk_size;
	} else if (x == n_chunks_x - 1 &&
			y == n_chunks_y - 1) { /* lower right */
		ubound_x = chunk_size;
		ubound_y = chunk_size;
	} else if (x == n_chunks_x - 1 && y != 0 &&
			y != n_chunks_y - 1) { /* flush right */
		ubound_x = chunk_size;
	} else if (x == n_chunks_x - 1 && y == 0) { /* upper right */
		ubound_x = chunk_size;
		lbound_y = 0;
	} else if (x != 0 && x != n_chunks_x - 1 && y == 0) { /* flush top */
		lbound_y = 0;
	}

	for (i = lbound_x; i < ubound_x; i++) {
		for (j = lbound_y; j < ubound_y; j++) {
			image_i = x * chunk_size/2 + i;
			image_j = y * chunk_size/2 + j;

			if ((image_i >= width) || (image_j >= height)) {
				continue;
			}

			chunk_index = j * chunk_size + i;
			image_index = image_j * width + image_i;

			normalized_output_image[c][image_index]
				= current[chunk_index];
		}
	}
}

/* write the resulting deconvoluted 16-bit TIFF image */
void output(char *output_image_filename)
{
	int i;

	printf("\nOutputting final image...\n");
	fflush(stdout);

	/* convert floats to uint16_t for tiff output */
	for (i = 0; i < 3 * width * height; i++) {
		/* earlier in init_images we added constant to prevent
		 * possible division by 0 caused by black pixels */
		normalized_output_image[i%3][i/3] -=
			DIV_BY_ZERO_PREVENTION;

		if (normalized_output_image[i%3][i/3] >= 1) {
			output_image[i] = UINT16_MAX;
		} else {
			output_image[i] = UINT16_MAX *
				normalized_output_image[i%3][i/3];
		}
	}

	write_tiff(output_image_filename, output_image, width, height);
}

/* also copies psf image over to opencl buffer */
void allocate_opencl_buffers()
{
	int i;

	for (i = 0; i < 3; i++) {
		k_image_a[i] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, chunk_size *
				chunk_size * sizeof(cl_float), NULL,
				NULL);
		k_image_b[i] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, chunk_size *
				chunk_size * sizeof(cl_float), NULL,
				NULL);
		k_original_image[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, chunk_size *
				chunk_size * sizeof(cl_float), NULL,
				NULL);
		k_psf_image[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, psf_width * psf_height
				* sizeof(cl_float), NULL, NULL);
		k_temp_image[i] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, chunk_size *
				chunk_size * sizeof(cl_float), NULL,
				NULL);

		clEnqueueWriteBuffer(queue, k_psf_image[i], CL_FALSE, 0,
				psf_width * psf_height * sizeof(cl_float),
				normalized_psf_image[i], 0, NULL,
				&copy_events[i][2]);
	}

}

void deconvolute_chunk(int chunk_index)
{
	int i;
	float *current;

	printf("Deconvoluting chunk: %d\n", chunk_index);
	fflush(stdout);

	/* copy image (RGB) into opencl buffer */
	for (i = 0; i < 3; i++) {
		current = chunks[chunk_index][i];

		clEnqueueWriteBuffer(queue, k_image_a[i], CL_TRUE, 0,
				chunk_size * chunk_size *
				sizeof(cl_float), current, 0, NULL,
				&copy_events[i][0]);
		clEnqueueWriteBuffer(queue, k_original_image[i],
				CL_FALSE, 0, chunk_size * chunk_size *
				sizeof(cl_float), current, 0, NULL,
				&copy_events[i][1]);

		clWaitForEvents(3, copy_events[i]);
	}

	/* run the deconvolution on this chunk */
	for (i = 0; i < n_iterations; i++) {
		do_iteration(i);
	}

	/* copy results back */
	for (i = 0; i < 3; i++) {
		current = chunks[chunk_index][i];

		clEnqueueReadBuffer(queue, k_image_a[i], CL_TRUE, 0,
				chunk_size * chunk_size *
				sizeof(cl_float), current, 0, NULL,
				NULL);
	}
}

void do_iteration(int i)
{
	int j;
	cl_mem *k_input_image;
	cl_mem *k_output_image;

	if (i%2 == 0) {
		k_input_image = k_image_a;
		k_output_image = k_image_b;
	} else {
		k_input_image = k_image_b;
		k_output_image = k_image_a;
	}

	/* convolution part */
	for (j = 0; j < 3; j++) {
		clSetKernelArg(convolution_kernel[j], 0, sizeof(cl_mem),
				&k_input_image[j]);
		clSetKernelArg(convolution_kernel[j], 1, sizeof(cl_mem),
				&k_psf_image[j]);
		clSetKernelArg(convolution_kernel[j], 2, sizeof(cl_mem),
				&k_temp_image[j]);
		clSetKernelArg(convolution_kernel[j], 3, sizeof(cl_int),
				&chunk_size);
		clSetKernelArg(convolution_kernel[j], 4, sizeof(cl_int),
				&chunk_size);
		clSetKernelArg(convolution_kernel[j], 5, sizeof(cl_int),
				&psf_width);
		clSetKernelArg(convolution_kernel[j], 6, sizeof(cl_int),
				&psf_height);

		clEnqueueNDRangeKernel(queue, convolution_kernel[j], 2,
				NULL, global_work_size, NULL, 0, NULL,
				&kernel_events[j]);
	}

	clWaitForEvents(3, kernel_events);

	/* deconvolution part */
	for (j = 0; j < 3; j++) {
		clSetKernelArg(deconvolution_kernel[j], 0,
				sizeof(cl_mem), &k_input_image[j]);
		clSetKernelArg(deconvolution_kernel[j], 1,
				sizeof(cl_mem), &k_psf_image[j]);
		clSetKernelArg(deconvolution_kernel[j], 2,
				sizeof(cl_mem), &k_output_image[j]);
		clSetKernelArg(deconvolution_kernel[j], 3,
				sizeof(cl_mem), &k_temp_image[j]);
		clSetKernelArg(deconvolution_kernel[j], 4,
				sizeof(cl_mem), &k_original_image[j]);
		clSetKernelArg(deconvolution_kernel[j], 5, sizeof(cl_int),
				&chunk_size);
		clSetKernelArg(deconvolution_kernel[j], 6, sizeof(cl_int),
				&chunk_size);
		clSetKernelArg(deconvolution_kernel[j], 7, sizeof(cl_int),
				&psf_width);
		clSetKernelArg(deconvolution_kernel[j], 8, sizeof(cl_int),
				&psf_height);

		clEnqueueNDRangeKernel(queue, deconvolution_kernel[j],
				2, NULL, global_work_size, NULL, 0,
				NULL, &kernel_events[j]);
	}

	clWaitForEvents(3, kernel_events);
}

/* free all the things */
void cleanup()
{
	int i, j;

	/* free images */
	for (i = 0; i < n_chunks_x * n_chunks_y; i++) {
		for (j = 0; j < 3; j++) {
			free(chunks[i][j]);
		}
		free(chunks[i]);
	}
	free(chunks);

	free(input_image);
	free(psf_image);
	free(output_image);
	for (i = 0; i < 3; i++) {
		free(normalized_input_image[i]);
		free(normalized_psf_image[i]);
		free(normalized_output_image[i]);
	}
	
	/* free opencl things */
	free(global_work_size);

	for (i = 0; i < 3; i++) {
		clReleaseKernel(convolution_kernel[i]);
		clReleaseKernel(deconvolution_kernel[i]);

		clReleaseMemObject(k_image_a[i]);
		clReleaseMemObject(k_image_b[i]);
		clReleaseMemObject(k_original_image[i]);
		clReleaseMemObject(k_psf_image[i]);
		clReleaseMemObject(k_temp_image[i]);
	}

	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

/* argv[1] = image to be deconvoluted, 16-bits per channel
   argv[2] = psf image, 8-bits per channel (due to GIMP limitations)
   argv[3] = number of iterations to run
 */
int main(int argc, char *argv[])
{
	int i;

	if (argc != 4) {
		fprintf(stderr, "Usage: deconvolute [input 16-bit TIFF image] [psf 8-bit TIFF image] [number of iterations]\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	/* force n_iterations to be even */
	n_iterations = atoi(argv[3]) + (atoi(argv[3]) % 2);

	/* get images, alloc memory, and split into chunks */
	init_images(argv[1], argv[2]);
	chunk_image();
	
	/* setup opencl stuffies */
	cl_utils_setup_gpu(&context, &queue, &device);
	program = cl_utils_create_program("deconvolute.cl", context, device);
	for (i = 0; i < 3; i++) {
		convolution_kernel[i] = clCreateKernel(program, "convolute", NULL);
		deconvolution_kernel[i] = clCreateKernel(program, "deconvolute", NULL);
	}

	/* alloc opencl memory and copy psf over */
	allocate_opencl_buffers();

	/* do the main deconvolution computations */
	global_work_size = emalloc(2 * sizeof(*global_work_size));
	global_work_size[0] = CHUNK_SIZE;
	global_work_size[1] = CHUNK_SIZE;
	printf("\n");
	for (i = 0; i < n_chunks_x * n_chunks_y; i++) {
		deconvolute_chunk(i);
	}

	/* unchunk the image */
	unchunk_image();

	/* output */
	output(OUT_FILENAME);

	cleanup();
	return 0;
}
