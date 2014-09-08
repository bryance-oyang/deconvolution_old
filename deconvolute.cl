__kernel void deconvolute(__global float *input_image, __global float
		*psf_image, __global float *output_image, __global float
		*temp_image, __global float *original_image, __global
		int *dimensions, __global int *psf_dimensions)
{
	int x, y;
	int i, j;
	int psf_i, psf_j;
	int psf_correction_x, psf_correction_y;
	int index, psf_index;
	int local_dim[2];
	int local_psf_dim[2];
	float total;

	x = get_global_id(0);
	y = get_global_id(1);

	local_dim[0] = dimensions[0];
	local_dim[1] = dimensions[1];
	local_psf_dim[0] = psf_dimensions[0];
	local_psf_dim[1] = psf_dimensions[1];

	psf_correction_x = psf_dimensions[0]/2;
	psf_correction_y = psf_dimensions[1]/2;

	total = 0;
	for (i = 0; i < local_dim[0]; i++) {
		for (j = 0; j < local_dim[1]; j++) {
			psf_i = i - x + psf_correction_x;
			psf_j = j - y + psf_correction_y;
			if ((psf_i >= local_psf_dim[0])
					|| (psf_j >= local_psf_dim[1])
					|| (psf_i < 0)
					|| (psf_j < 0)) {
				continue;
			}
			
			index = j * local_dim[0] + i;
			psf_index = psf_j * local_psf_dim[0] + psf_i;
			total += psf_image[psf_index] *
				original_image[index] /
				temp_image[index];
		}
	}

	total *= input_image[y * local_dim[0] + x];

	output_image[y * local_dim[0] + x] = total;
}

__kernel void convolute(__global float *input_image, __global float
		*psf_image, __global float *temp_image, __global int
		*dimensions, __global int *psf_dimensions)
{
	int x, y;
	int i, j;
	int psf_i, psf_j;
	int psf_correction_x, psf_correction_y;
	int index, psf_index;
	int local_dim[2];
	int local_psf_dim[2];
	float total;

	x = get_global_id(0);
	y = get_global_id(1);

	local_dim[0] = dimensions[0];
	local_dim[1] = dimensions[1];
	local_psf_dim[0] = psf_dimensions[0];
	local_psf_dim[1] = psf_dimensions[1];

	psf_correction_x = psf_dimensions[0]/2;
	psf_correction_y = psf_dimensions[1]/2;

	total = 0;
	for (i = 0; i < local_dim[0]; i++) {
		for (j = 0; j < local_dim[1]; j++) {
			psf_i = x - i + psf_correction_x;
			psf_j = y - j + psf_correction_y;
			if ((psf_i >= local_psf_dim[0])
					|| (psf_j >= local_psf_dim[1])
					|| (psf_i < 0)
					|| (psf_j < 0)) {
				continue;
			}

			index = j * local_dim[0] + i;
			psf_index = psf_j * local_psf_dim[0] + psf_i;
			total += input_image[index] *
				psf_image[psf_index];
		}
	}

	temp_image[y * local_dim[0] + x] = total;
}
