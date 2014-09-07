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

/* free all the things */
void cleanup()
{
	int i;

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
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

/* argv[1] = image to be deconvoluted, 16-bits per channel
   argv[2] = psf image, 8-bits per channel (due to GIMP limitations)
 */
int main(int argc, char *argv[])
{
	if (argc != 3) {
		fprintf(stderr, "Usage: deconvolute [input 16-bit TIFF image] [psf 8-bit TIFF image]\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	init_images(argv[1], argv[2]);
	
	cl_utils_setup_gpu(&context, &queue);
	program = cl_utils_create_program("deconvolute.cl", context);

	output(OUT_FILENAME);

	cleanup();
	return 0;
}
