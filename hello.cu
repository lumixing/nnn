#include <stdio.h>

__global__ void mat_add(float *a, float *b, float *c) {
	// int i = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.x;
	if (i >= 6) return;
	c[i] = a[i] + b[i];
}

int main() {
	float *a;  // 2x3
	float *b;
	float *c;
	cudaMallocManaged(&a, 6*sizeof(float));
	cudaMallocManaged(&b, 6*sizeof(float));
	cudaMallocManaged(&c, 6*sizeof(float));
	a[0] = 0; a[1] = 1; a[2] = 2;
	a[3] = 3; a[4] = 4; a[5] = 5;
	b[0] = 6; b[1] = 7; b[2] = 8;
	b[3] = 9; b[4] = 10; b[5] = 11;

	mat_add<<<1, 6>>>(a, b, c);
	cudaDeviceSynchronize();

	printf("%f %f %f\n", c[0], c[1], c[2]);
	printf("%f %f %f\n", c[3], c[4], c[5]);

	cudaFree(&a);
	cudaFree(&b);
	cudaFree(&c);
	return 0;
}
