#include <stdio.h>

#define MATLEN(m) (m.rows*m.cols)
#define MATGET(m, r, c) (m.data[r*m.cols+c])
#define MATDATAGET(md, r, c, mc) (md[r*mc+c])
#define VEC 1

#define THREADS 256
#define MATCHE(m) ((MATLEN(m)+THREADS-1)/THREADS),256

typedef struct {
	float *data;
	int rows, cols;
} mat;

mat mat_make(int rows, int cols) {
	mat c;
	cudaMallocManaged(&c.data, rows*cols*sizeof(float));
	c.rows = rows;
	c.cols = cols;
	return c;
}

float randf() {
	return (float)rand() / (float)RAND_MAX;
}

mat mat_make_random(int rows, int cols) {
	mat c = mat_make(rows, cols);
	float scale = sqrtf(2./float(rows));
	for (int i = 0; i < MATLEN(c); i++) {
		c.data[i] = (randf()*2-1)*scale;
	}
	return c;
}

void mat_copy(mat src, mat dst) {
	// for (int i = 0; i < MATLEN(src); i++) {
	// 	dst.data[i] = src.data[i];
	// }
	cudaMemcpy(dst.data, src.data, MATLEN(src)*sizeof(float), cudaMemcpyDeviceToDevice);
}

__device__ float act(float x) {  // leaky relu
	return x > 0 ? x : 0.01 * x;
}

__global__ void kmat_act(float *a, int s) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= s) return;

	a[i] = act(a[i]);
}

void mat_act(mat a) {
	kmat_act<<<MATCHE(a)>>>(a.data, MATLEN(a));
}

__global__ void kmat_softmax_max(float *a, int s, float *out) {
	float max = a[0];
	for (int i = 0; i < s; i++) {
		max = fmaxf(max, a[i]);
	}
	*out = max;
}

__global__ void kmat_softmax_sum(float *a, int s, float *max, float *out) {
	float sum = 0;
	for (int i = 0; i < s; i++) {
		a[i] = expf(a[i] - *max);
		sum += a[i];
	}
	*out = sum;
}

__global__ void kmat_softmax_div(float *a, int s, float *sum) {
	for (int i = 0; i < s; i++) {
		a[i] /= *sum;
	}
}

void mat_softmax(mat a, float *max, float *sum) {
	// float max = a.data[0];
	// for (int i = 0; i < MATLEN(a); i++) {
	// 	max = fmaxf(max, a.data[i]);
	// }
	kmat_softmax_max<<<1>>>(a.data, MATLEN(a), max);
	
	// float sum = 0;
	// for (int i = 0; i < MATLEN(a); i++) {
	// 	a.data[i] = expf(a.data[i] - max);
	// 	sum += a.data[i];
	// }
	kmat_softmax_sum<<<1>>>(a.data, MATLEN(a), max, sum);

	// for (int i = 0; i < MATLEN(a); i++) {
	// 	a.data[i] /= sum;
	// }
	kmat_softmax_div<<<1>>>(a.data, MATLEN(a), sum);
}

__global__ void kmat_add(float *a, float *b, float *c, int s) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= s) return;

	c[i] = a[i] + b[i];
}

void mat_add(mat a, mat b, mat c) {
	kmat_add<<<MATCHE(a)>>>(a.data, b.data, c.data, MATLEN(a));
}

__global__ void kmat_dot(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= a_rows) return;
	if (j >= b_cols) return;

	float sum = 0;
	for (int k = 0; k < a_cols; k++) {
		sum += a[a_cols*i+k] * b[b_cols*k+j];
	}
	c[b_cols*i+j] = sum;
}

void mat_dot(mat a, mat b, mat c) {
	dim3 block(16, 16);
	dim3 grid((b.cols + 15) / 16, (a.rows + 15) / 16);
	kmat_dot<<<grid, block>>>(a.data, b.data, c.data, a.rows, a.cols, b.cols);
}
