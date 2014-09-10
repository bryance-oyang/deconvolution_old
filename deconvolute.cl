__kernel void deconvolute(__global float *input_image, __global float
		*psf_image, __global float *output_image, __global float
		*temp_image, __global float *original_image, int width,
		int height, int psf_width, int psf_height)
{
	int x, y;
	int i, j;
	int psf_i, psf_j;
	int psf_correction_x, psf_correction_y;
	int index, psf_index;
	float total;

	x = get_global_id(0);
	y = get_global_id(1);

	psf_correction_x = psf_width/2;
	psf_correction_y = psf_height/2;

	total = 0;
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			psf_i = i - x + psf_correction_x;
			psf_j = j - y + psf_correction_y;
			if ((psf_i >= psf_width)
					|| (psf_j >= psf_height)
					|| (psf_i < 0)
					|| (psf_j < 0)) {
				continue;
			}
			
			index = j * width + i;
			if (temp_image[index] == 0)
				continue;

			psf_index = psf_j * psf_width + psf_i;
			total += psf_image[psf_index] *
				original_image[index] /
				temp_image[index];
		}
	}

	total *= input_image[y * width + x];

	output_image[y * width + x] = total;
}

__kernel void convolute(__global float *input_image, __global float
		*psf_image, __global float *temp_image,
		int width, int height, int psf_width, int psf_height)
{
	int x, y;
	int i, j;
	int psf_i, psf_j;
	int psf_correction_x, psf_correction_y;
	int index, psf_index;
	float total;

	x = get_global_id(0);
	y = get_global_id(1);

	psf_correction_x = psf_width/2;
	psf_correction_y = psf_height/2;

	total = 0;
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			psf_i = x - i + psf_correction_x;
			psf_j = y - j + psf_correction_y;
			if ((psf_i >= psf_width)
					|| (psf_j >= psf_height)
					|| (psf_i < 0)
					|| (psf_j < 0)) {
				continue;
			}

			index = j * width + i;
			psf_index = psf_j * psf_width + psf_i;
			total += input_image[index] *
				psf_image[psf_index];
		}
	}

	temp_image[y * width + x] = total;
}
