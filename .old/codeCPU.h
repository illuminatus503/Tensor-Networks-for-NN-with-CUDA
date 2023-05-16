double timing_CPU(struct timespec begin, struct timespec end);
void init_vectors(float *A, float *B, unsigned int N);
double add_vectors_CPU(float *A, float *B, float *C, unsigned int N);
double prod_vectors_CPU(float *A, float *B, float *C, unsigned int N);
double dot_prod_vectors_CPU(float *A, float *B, float *C, unsigned int N);
void print_vector(float *C, unsigned int N);
