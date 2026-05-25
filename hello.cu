#include <stdlib.h>
#include <assert.h>
#include "mat.cu"

#define INPUTS 28*28
#define OUTPUTS 10
#define IMGS 60000
#define BATCH 256


typedef struct {
	mat w, b, z, l;
	mat dw, db, dz, dl;
} Layer;

Layer make_layer(int inputs, int outputs) {
	Layer l;
	l.w  = mat_make_random(inputs, outputs);
	l.b  = mat_make(1, outputs);
	l.z  = mat_make(BATCH, outputs);
	l.l  = mat_make(BATCH, outputs);
	l.dw = mat_make(inputs, outputs);
	l.db = mat_make(1, outputs);
	l.dz = mat_make(BATCH, outputs);
	l.dl = mat_make(BATCH, outputs);
	return l;
}

char *read_file(const char *path, long *out_size);
void mat_calc(mat a, Layer l1, Layer l2, Layer l3, float *max, float *sum);

__global__ void kmat_sum_rows(float *dz, float *db, int rows, int cols) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= cols) return;
    float sum = 0;
    for (int i = 0; i < rows; i++) {
        sum += dz[i*cols + j];
    }
    db[j] = sum;
}

void mat_sum_rows(mat dz, mat db) {
    kmat_sum_rows<<<(dz.cols+255)/256, 256>>>(dz.data, db.data, dz.rows, dz.cols);
}

void layer_dlast(Layer l, Layer lprev, mat target) {
	mat_sub(l.l, target, l.dl);
	mat_copy(l.dl, l.dz);
	mat_dot(mat_transpose(lprev.l), l.dz, l.dw);
	// mat_copy(l.dz, l.db);
	mat_sum_rows(l.dz, l.db);
}

__device__ float relu_derive(float x) {
    return x > 0 ? 1 : 0.01;
}

__global__ void layer_d_actder(float *dz, float *dl, float *z, int s) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= s) return;
	dz[i] = dl[i] * relu_derive(z[i]);
}

void layer_d(Layer l, Layer lnext, mat input) {
	mat_dot(lnext.dz, mat_transpose(lnext.w), l.dl);
	layer_d_actder<<<MATCHE(l.dz)>>>(l.dz.data, l.dl.data, l.z.data, MATLEN(l.dz));
	mat_dot(mat_transpose(input), l.dz, l.dw);
	// mat_copy(l.dz, l.db);
	mat_sum_rows(l.dz, l.db);
}

void layer_learn(Layer l) {
	mat_scale(l.dw, LR, l.dw);
	mat_scale(l.db, LR, l.db);
	mat_sub(l.w, l.dw, l.w);
	mat_sub(l.b, l.db, l.b);
}

cublasHandle_t handle;

int main() {
	srand(time(NULL));
	cublasCreate(&handle);

	// todo: skip header initially not later
	long images_size;
	char *images_data = read_file("train-images.idx3-ubyte", &images_size);
	images_data += 16;
	images_size -= 16;
	long inputs_size = images_size;
	printf("inputs_size=%ld\n", inputs_size);
	float *inputs;
	cudaMallocManaged(&inputs, inputs_size*sizeof(float));
	for (int i = 0; i < inputs_size; i++) {
		inputs[i] = (float)(images_data[i])/255.;
	}
	free(images_data-16);

	long labels_size;
	char *labels_data = read_file("train-labels.idx1-ubyte", &labels_size);
	labels_data += 8;
	labels_size -= 8;
	long outputs_size = OUTPUTS*labels_size;
	printf("outputs_size=%ld\n", outputs_size);
	float *outputs;
	cudaMallocManaged(&outputs, outputs_size*sizeof(float));
	cudaMemset(outputs, 0, outputs_size*sizeof(float));
	for (int i = 0; i < labels_size; i++) {
		outputs[i*10+labels_data[i]] = 1;
	}
	free(labels_data-8);

	printf("Making layers...\n");
	Layer l1 = make_layer(INPUTS, 128);
	Layer l2 = make_layer(128, 64);
	Layer l3 = make_layer(64, OUTPUTS);

	mat input = mat_make(BATCH, INPUTS);
	mat target = mat_make(BATCH, OUTPUTS);

	// mat a = mat_make(2, 3);
	// mat b = mat_make(3, 2);
	// mat c = mat_make(2, 2);
	// a.data[0] = 0; a.data[1] = 1; a.data[2] = 2;
	// a.data[3] = 3; a.data[4] = 4; a.data[5] = 5;
	// b.data[0] = 6; b.data[1] = 7; b.data[2] = 8;
	// b.data[3] = 9; b.data[4] = 10; b.data[5] = 11;
	// mat_dot(a, b, c);
	// cudaDeviceSynchronize();
	// printf("%f %f\n", c.data[0], c.data[1]);
	// printf("%f %f\n", c.data[2], c.data[3]);

	// return;

	float *max;
	float *sum;
	cudaMallocManaged(&max, sizeof(float));
	cudaMallocManaged(&sum, sizeof(float));

	for (int epoch = 0; epoch < 100; epoch++) {
		printf("Epoch %d\n", epoch);
		for (int tr_idx = 0; tr_idx < IMGS; tr_idx += BATCH) {
			// if (tr_idx % 1000 == 0) {
			// 	printf("%d/%d\n", tr_idx, IMGS);
			// }

			input.data = inputs+(28*28*tr_idx);
			mat_calc(input, l1, l2, l3, max, sum);

			target.data = outputs+(tr_idx*10);

			layer_dlast(l3, l2, target);
			layer_d(l2, l3, l1.l);
			layer_d(l1, l2, input);

			layer_learn(l1);
			layer_learn(l2);
			layer_learn(l3);
		}

		// cudaDeviceSynchronize();

		{
			int correct = 0;
			for (int i = 0; i < 1000; i++) {
				input.data = inputs+(28*28*i);
				input.rows = 1;  // single image!!
				mat_calc(input, l1, l2, l3, max, sum);
				cudaDeviceSynchronize();

				// argmax of output
				int predicted = 0;
				for (int j = 1; j < 10; j++) {
					if (l3.l.data[j] > l3.l.data[predicted]) predicted = j;
				}

				// argmax of label
				int actual = 0;
				for (int j = 1; j < 10; j++) {
					if (outputs[i*10+j] > outputs[i*10+actual]) actual = j;
				}

				if (predicted == actual) correct++;
			}
			printf("accuracy: %.1f%%\n", correct/10.0f);
			input.rows = BATCH;  // restore batch size!!
		}
	}

	printf("Done!\n");
	return 0;
}

char *read_file(const char *path, long *out_size) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;

	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);

	char *buf = (char*)malloc(size);
	fread(buf, 1, size, f);
	fclose(f);

	if (out_size) *out_size = size;
	return buf;
}

void layer(mat a, Layer l) {
	// 256x784 * 784x128 = 256x128
	// 256x128 * 128x64 = 256x64
	mat_dot(a, l.w, l.l);
	// 256x128 + 256x128 = 256x128
	// mat_add(l.l, l.b, l.l);
	kmat_add_broadcast<<<MATCHE(l.l)>>>(l.l.data, l.b.data, l.l.data, l.l.rows, l.l.cols);
	// 256x128 = 256x128
	mat_copy(l.l, l.z);
	// 256x128 = 256x128
	// 256x64
	mat_act(l.l);
}

void layer2(mat a, Layer l, float *max, float *sum) {
	// 256x64 * 64x10 = 256x10
	mat_dot(a, l.w, l.l);
	// mat_add(l.l, l.b, l.l);
	kmat_add_broadcast<<<MATCHE(l.l)>>>(l.l.data, l.b.data, l.l.data, l.l.rows, l.l.cols);
	// mat_copy(l.l, l.z);
	// 256x10
	mat_softmax2(l.l, max, sum);
}

void mat_calc(mat a, Layer l1, Layer l2, Layer l3, float *max, float *sum) {
	layer(a, l1);
	layer(l1.l, l2);
	layer2(l2.l, l3, max, sum);
}

// int main2() {
// 	float *a;  // 2x3
// 	float *b;
// 	float *c;
// 	cudaMallocManaged(&a, 6*sizeof(float));
// 	cudaMallocManaged(&b, 6*sizeof(float));
// 	cudaMallocManaged(&c, 6*sizeof(float));
// 	a[0] = 0; a[1] = 1; a[2] = 2;
// 	a[3] = 3; a[4] = 4; a[5] = 5;
// 	b[0] = 6; b[1] = 7; b[2] = 8;
// 	b[3] = 9; b[4] = 10; b[5] = 11;

// 	mat_add<<<1, 6>>>(a, b, c);
// 	cudaDeviceSynchronize();

// 	// printf("%f %f %f\n", c[0], c[1], c[2]);
// 	// printf("%f %f %f\n", c[3], c[4], c[5]);

// 	cudaFree(&a);
// 	cudaFree(&b);
// 	cudaFree(&c);
// 	return 0;
// }
