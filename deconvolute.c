#include "deconvolute.h"

/* malloc all images and read in input image and psf
   also normalizes the psf with the variable total[3]
   by channel
 */
void init_images(char *input_image_filename, char *psf_image_filename)
{
	int i;

	input_image = read_tiff(input_image_filename, &width, &height);
	psf_image = read_tiff8(psf_image_filename, &psf_width, &psf_height);
	output_image = emalloc(3 * width * height *
			sizeof(*output_image));

	normalized_input_image = emalloc(3 *
			sizeof(*normalized_input_image));
	normalized_psf_image = emalloc(3 *
			sizeof(*normalized_psf_image));
	normalized_output_image = emalloc(3 *
			sizeof(*normalized_output_image));
	for (i = 0; i < 3; i++) {
		normalized_input_image[i] = emalloc(width * height *
				sizeof(*(normalized_input_image[i])));
		normalized_psf_image[i] = emalloc(psf_width * psf_height
				* sizeof(*(normalized_psf_image[i])));
		normalized_output_image[i] = emalloc(width * height *
				sizeof(*(normalized_output_image[i])));
	}

	for (i = 0; i < 3 * width * height; i++) {
		normalized_input_image[i%3][i/3] =
			(float)input_image[i]/UINT16_MAX;
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

/* write the resulting deconvoluted 16-bit TIFF image */
void output(char *output_image_filename)
{
	int i;

	/* retrieve result from opencl memory */
	for (i = 0; i < 3; i++) {
		clEnqueueReadBuffer(queue, k_image_a[i], CL_TRUE, 0,
				width * height
				* sizeof(cl_float),
				normalized_output_image[i], 0,
				NULL, NULL);
	}

	/* convert floats to uint16_t for tiff output */
	for (i = 0; i < 3 * width * height; i++) {
		if (normalized_output_image[i%3][i/3] >= 1) {
			output_image[i] = UINT16_MAX;
		} else {
			output_image[i] = UINT16_MAX *
				normalized_output_image[i%3][i/3];
		}
	}

	write_tiff(output_image_filename, output_image, width, height);
}

void copy_images_to_opencl()
{
	int i;

	dimensions = emalloc(2 * sizeof(*dimensions));
	psf_dimensions = emalloc(2 * sizeof(*psf_dimensions));

	dimensions[0] = width;
	dimensions[1] = height;
	psf_dimensions[0] = psf_width;
	psf_dimensions[1] = psf_height;

	for (i = 0; i < 3; i++) {
		k_image_a[i] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, width * height *
				sizeof(cl_float), NULL, NULL);
		k_image_b[i] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, width * height *
				sizeof(cl_float), NULL, NULL);
		k_original_image[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, width * height *
				sizeof(cl_float), NULL, NULL);
		k_psf_image[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, psf_width * psf_height
				* sizeof(cl_float), NULL, NULL);
		k_temp_image[i] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, width * height *
				sizeof(cl_float), NULL, NULL);
		k_dimensions[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, 2 * sizeof(cl_int),
				NULL, NULL);
		k_psf_dimensions[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, 2 * sizeof(cl_int),
				NULL, NULL);

		clEnqueueWriteBuffer(queue, k_image_a[i], CL_TRUE, 0,
				width * height * sizeof(cl_float),
				normalized_input_image[i], 0, NULL,
				&copy_events[i][0]);
		clEnqueueWriteBuffer(queue, k_original_image[i],
				CL_FALSE, 0, width * height *
				sizeof(cl_float),
				normalized_input_image[i], 0, NULL,
				&copy_events[i][1]);
		clEnqueueWriteBuffer(queue, k_psf_image[i], CL_FALSE, 0,
				psf_width * psf_height * sizeof(cl_float),
				normalized_psf_image[i], 0, NULL,
				&copy_events[i][2]);
		clEnqueueWriteBuffer(queue, k_dimensions[i], CL_FALSE,
				0, 2 * sizeof(cl_int), dimensions, 0,
				NULL, &copy_events[i][3]);
		clEnqueueWriteBuffer(queue, k_psf_dimensions[i], CL_FALSE,
				0, 2 * sizeof(cl_int), psf_dimensions, 0,
				NULL, &copy_events[i][4]);
		clWaitForEvents(5, copy_events[i]);
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
		clSetKernelArg(convolution_kernel[j], 3, sizeof(cl_mem),
				&k_dimensions[j]);
		clSetKernelArg(convolution_kernel[j], 4, sizeof(cl_mem),
				&k_psf_dimensions[j]);

		if (i == 0) {
			clEnqueueNDRangeKernel(queue,
					convolution_kernel[j], 2, NULL,
					global_work_size, NULL, 0, NULL,
					&kernel_events[j]);
		} else {
			clEnqueueNDRangeKernel(queue,
					convolution_kernel[j], 2, NULL,
					global_work_size, NULL, 0, NULL,
					&kernel_events[j]);
		}
	}

	clWaitForEvents(3, kernel_events);

	/* deconvolution part */
	for (j = 0; j < 3; j++) {
		clSetKernelArg(deconvolution_kernel[j], 0,
				sizeof(cl_mem), &k_input_image);
		clSetKernelArg(deconvolution_kernel[j], 1,
				sizeof(cl_mem), &k_psf_image);
		clSetKernelArg(deconvolution_kernel[j], 2,
				sizeof(cl_mem), &k_output_image);
		clSetKernelArg(deconvolution_kernel[j], 3,
				sizeof(cl_mem), &k_temp_image);
		clSetKernelArg(deconvolution_kernel[j], 4,
				sizeof(cl_mem), &k_original_image);
		clSetKernelArg(deconvolution_kernel[j], 5,
				sizeof(cl_mem), &k_dimensions);
		clSetKernelArg(deconvolution_kernel[j], 6,
				sizeof(cl_mem), &k_psf_dimensions);

		clEnqueueNDRangeKernel(queue, deconvolution_kernel[j],
				2, NULL, global_work_size, NULL, 0,
				NULL, &kernel_events[j]);
	}

	clWaitForEvents(3, kernel_events);
}

/* free all the things */
void cleanup()
{
	int i;

	/* free images */
	free(input_image);
	free(psf_image);
	free(output_image);
	for (i = 0; i < 3; i++) {
		free(normalized_input_image[i]);
		free(normalized_psf_image[i]);
		free(normalized_output_image[i]);
	}
	free(normalized_input_image);
	free(normalized_psf_image);
	free(normalized_output_image);
	
	free(dimensions);
	free(psf_dimensions);

	/* free opencl things */
	free(global_work_size);

	for (i = 0; i < 3; i++) {
		clReleaseKernel(convolution_kernel[i]);
		clReleaseKernel(deconvolution_kernel[i]);

		clReleaseMemObject(k_image_a[i]);
		clReleaseMemObject(k_image_b[i]);
		clReleaseMemObject(k_psf_image[i]);
		clReleaseMemObject(k_temp_image[i]);
		clReleaseMemObject(k_dimensions[i]);
		clReleaseMemObject(k_psf_dimensions[i]);
	}

	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

/* argv[1] = image to be deconvoluted, 16-bits per channel
   argv[2] = psf image, 8-bits per channel (due to GIMP limitations)
 */
int main(int argc, char *argv[])
{
	int i;
	int n_iterations;

	if (argc != 3) {
		fprintf(stderr, "Usage: deconvolute [input 16-bit TIFF image] [psf 8-bit TIFF image]\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	/* make sure n_iterations is even */
	n_iterations = N_ITERATIONS + (N_ITERATIONS % 2);

	/* get images and alloc memory */
	init_images(argv[1], argv[2]);
	
	/* setup opencl stuffies */
	cl_utils_setup_gpu(&context, &queue, &device);
	program = cl_utils_create_program("deconvolute.cl", context,
			device);
	for (i = 0; i < 3; i++) {
		convolution_kernel[i] = clCreateKernel(program, "convolute", NULL);
		deconvolution_kernel[i] = clCreateKernel(program, "deconvolute", NULL);
	}

	/* alloc opencl memory and copy images over */
	copy_images_to_opencl();

	/* do the main deconvolution computations */
	global_work_size = emalloc(2 * sizeof(*global_work_size));
	global_work_size[0] = width;
	global_work_size[1] = height;
	for (i = 0; i < n_iterations; i++) {
		do_iteration(i);
		printf("Pass %d completed\n", i);
		fflush(stdout);
	}

	/* output */
	output(OUT_FILENAME);

	cleanup();
	return 0;
}
