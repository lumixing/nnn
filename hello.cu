#include <stdlib.h>
#include <assert.h>
#include "mat.cu"

#define INPUTS 28*28
#define OUTPUTS 10
#define IMGS 60000

typedef struct {
	mat w, b, z, l;
	mat dw, db, dz, dl;
} Layer;

Layer make_layer(int inputs, int outputs) {
	Layer l;
	l.w  = mat_make_random(inputs, outputs);
	l.b  = mat_make(VEC, outputs);
	l.z  = mat_make(VEC, outputs);
	l.l  = mat_make(VEC, outputs);
	l.dw = mat_make(inputs, outputs);
	l.db = mat_make(VEC, outputs);
	l.dz = mat_make(VEC, outputs);
	l.dl = mat_make(VEC, outputs);
	return l;
}

char *read_file(const char *path, long *out_size);
void mat_calc(mat a, Layer l1, Layer l2, Layer l3, float *max, float *sum);

int main() {
	srand(time(NULL));

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

	mat input = mat_make(VEC, INPUTS);
	mat target = mat_make(VEC, OUTPUTS);

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

	float *max;
	float *sum;
	cudaMallocManaged(&max, sizeof(float));
	cudaMallocManaged(&sum, sizeof(float));

	for (int epoch = 0; epoch < 100; epoch++) {
		printf("Epoch %d\n", epoch);
		for (int tr_idx = 0; tr_idx < IMGS; tr_idx++) {
			if (tr_idx % 1000 == 0) {
				printf("%d/%d\n", tr_idx, IMGS);
			}

			input.data = inputs+(28*28*tr_idx);
			mat_calc(input, l1, l2, l3, max, sum);
			cudaDeviceSynchronize();
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
	mat_dot(a, l.w, l.l);
	mat_add(l.l, l.b, l.l);
	mat_copy(l.l, l.z);
	mat_act(l.l);
}

void layer2(mat a, Layer l, float *max, float *sum) {
	mat_dot(a, l.w, l.l);
	mat_add(l.l, l.b, l.l);
	mat_copy(l.l, l.z);
	mat_softmax(l.l, max, sum);
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
