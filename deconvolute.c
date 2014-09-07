int main(int argc, char *argv[])
{
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	
	cl_utils_setup_gpu(&context, &queue);
	program = cl_utils_create_program("deconvolute.cl", context);
}
