#include "opencl_utils.h"

/* reads a file and returns a malloced char* of its contents
   that needs to freed */
char *cl_utils_read_file(char *filename)
{
	FILE *file;
	int size;
	char *contents;

	if ((file = fopen(filename, "r")) == NULL)
		return NULL;
	if (fseek(file, 0, SEEK_END) == -1)
		return NULL;
	if ((size = ftell(file)) == -1)
		return NULL;
	if (fseek(file, 0, SEEK_SET) == -1)
		return NULL;

	contents = emalloc(size + 1);

	fread(contents, 1, size, file);
	if (ferror(file)) {
		free(contents);
		return NULL;
	}

	fclose(file);
	contents[size] = '\0';
	return contents;
}

/* creates opencl context and command queue using gpu device */
void cl_utils_setup_gpu(cl_context *context, cl_command_queue
		*command_queue)
{
	cl_platform_id platform;
	cl_device_id device;

	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	*context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
	*command_queue = clCreateCommandQueue(*context, device, 0,
			NULL);
}

/* create a opencl program from source code in filename */
cl_program cl_utils_create_program(char *filename, cl_context context)
{
	cl_int err;
	cl_program program;
	char *source_code;

	if ((source_code = cl_utils_read_file(filename)) == NULL) {
		fprintf(stderr, "cl_utils_create_program: could not read %s\n",
				filename);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	program = clCreateProgramWithSource(context, 1,
			(const char **) &source_code, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource failed\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clBuildProgram failed\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	free(source_code);
	return program;
}
