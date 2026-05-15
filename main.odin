// #+vet explicit-allocators
package main

import "core:slice"
import "core:time"
import "core:os"
import "core:math"
import "core:fmt"

D  :: 0.001
LR :: 0.001  // learning rate

VEC :: 1

INPUTS :: 28*28
// MIDDLES :: 2
OUTPUTS :: 10

Layer :: struct {
	w, b:   Matrix,
	z, l:   Matrix,
	dw, db: Matrix,
	dz, dl: Matrix,
}

make_layer :: proc(inputs, outputs: int) -> Layer {
	return {
		w  = mat_make_random(inputs, outputs),
		b  = mat_make(VEC, outputs),
		z  = mat_make(VEC, outputs),
		l  = mat_make(VEC, outputs),
		dw = mat_make(inputs, outputs),
		db = mat_make(VEC, outputs),
		dz = mat_make(VEC, outputs),
		dl = mat_make(VEC, outputs),
	}
}

layer_dlast :: proc(l, lprev: ^Layer, target: Matrix) #no_bounds_check {
	// dl = output - target  (softmax + cross entropy derivative)
	mat_sub(l.l, target, l.dl)
	// dz = dl  (softmax absorbs the sigmoid step, so dz = dl)
	mat_copy(l.dl, l.dz)
	// dw = lprev.lᵀ · dz
	mat_mul(mat_transpose(lprev.l), l.dz, l.dw)
	// db = dz
	mat_copy(l.dz, l.db)
}

layer_d :: proc(l, lnext: ^Layer, input: Matrix) #no_bounds_check {
	// dl = lnext.dz · lnext.wᵀ
	mat_mul(lnext.dz, mat_transpose(lnext.w), l.dl)
	// dz = dl ⊙ relu'(l)   relu'(x) = 1 if x>0 else 0
	for i in 0..<l.dz.cols {
		l.dz.data[i] = l.dl.data[i] * relu_derive(l.z.data[i])
	}
	// dw = inputᵀ · dz
	mat_mul(mat_transpose(input), l.dz, l.dw)
	// db = dz
	mat_copy(l.dz, l.db)
}

layer_learn :: proc(l: ^Layer) #no_bounds_check {
	mat_scale(l.dw, LR, l.dw)
	mat_scale(l.db, LR, l.db)
	mat_sub(l.w, l.dw, l.w)
	mat_sub(l.b, l.db, l.b)
}

IMGS :: 60_000
IMGSZ :: 28*28

MAXXX :: 1000*100

imgdata: []f32
lbldata: []f32

main :: proc() {
	rawimgdata, err := os.read_entire_file("train-images.idx3-ubyte", context.allocator)
	assert(err == nil)
	imgdata = make([]f32, IMGSZ*IMGS)
	for d, idx in rawimgdata[16:] {
		imgdata[idx] = f32(d)/255.
	}
	delete(rawimgdata)

	rawlbldata, err2 := os.read_entire_file("train-labels.idx1-ubyte", context.allocator)
	assert(err2 == nil)
	lbldata = make([]f32, 10*IMGS)
	for d, idx in rawlbldata[8:] {
		lbldata[idx*10+int(d)] = 1.
	}
	
	delete(rawlbldata)

	// a := mat_make(VEC, INPUTS)

	// 784 → 128 → 64 → 10
	l1 := make_layer(28*28, 128)
	l2 := make_layer(128, 64)
	l3 := make_layer(64, 10)

	target := mat_make_from(l3.l)
	input := mat_make(VEC, INPUTS)

	defer {
		sz := l1.w.rows*l1.w.cols + l1.b.rows*l1.b.cols
		sz += l2.w.rows*l2.w.cols + l2.b.rows*l2.b.cols
		sz += l3.w.rows*l3.w.cols + l3.b.rows*l3.b.cols
		fmt.println("dumping weights and biases...", sz, sz*size_of(f32))
		defer fmt.println("Done!!!")
		d := make([]f32, sz)

		idx := 0
		idx += copy(d[idx:], l1.w.data)
		idx += copy(d[idx:], l1.b.data)
		idx += copy(d[idx:], l2.w.data)
		idx += copy(d[idx:], l2.b.data)
		idx += copy(d[idx:], l3.w.data)
		idx += copy(d[idx:], l3.b.data)

		// omg := cast([^]u8)raw_data(d)
		// omg2 := slice.from_ptr(omg, sz*size_of(f32))
		err := os.write_entire_file("wb.f32", slice.to_bytes(d))
		assert(err == nil)
	}

	tt := time.now()
	#no_bounds_check for epoch in 0..<60 {
		if true {
			correct: int
			for i in 0..<1000 {  // test on first 1000 images
				input.data = imgdata[28*28*i:][:28*28]
				mat_calc(input, l1, l2, l3)
				predicted := argmax(l3.l)
				actual := argmax({data=lbldata[i*10:][:10], rows=1, cols=10})
				if predicted == actual do correct += 1
			}
			fmt.printfln("epoch %v accuracy: %.1f%%", epoch, f32(correct)/10.0)
		}

		for tr_idx in 0..<IMGS {
			if tr_idx%5000==0 {
				fmt.println(tr_idx,IMGS)
			}

			input.data = imgdata[28*28*tr_idx:][:28*28]
			mat_calc(input, l1, l2, l3)

			target.data = lbldata[tr_idx*10:][:10]

			layer_dlast(&l3, &l2, target)
			layer_d(&l2, &l3, l1.l)
			layer_d(&l1, &l2, input)

			layer_learn(&l1)
			layer_learn(&l2)
			layer_learn(&l3)
		}
	}

	fmt.println("Finished in", time.since(tt))
}

layer :: proc(a, w, b, z, l: Matrix) #no_bounds_check {
	// assert_eq(l.rows, b.rows); assert_eq(l.cols, b.cols)

	mat_mul(a, w, l)
	mat_add(l, b, l)
	mat_copy(l, z)
	mat_act(l)
	return
}

layer2 :: proc(a, w, b, z, l: Matrix) #no_bounds_check {
	// assert_eq(l.rows, b.rows); assert_eq(l.cols, b.cols)

	mat_mul(a, w, l)
	mat_add(l, b, l)
	mat_copy(l, z)
	// mat_act(l)
	softmax(l)
	return
}

mat_calc :: proc(a: Matrix, l1, l2, l3: Layer) #no_bounds_check {
	// assert_eq(l1.l.rows, l1.b.rows); assert_eq(l1.l.cols, l1.b.cols)
	// assert_eq(l2.l.rows, l2.b.rows); assert_eq(l2.l.cols, l2.b.cols)
	// assert_eq(l3.l.rows, l3.b.rows); assert_eq(l3.l.cols, l3.b.cols)

	layer(a, l1.w, l1.b, l1.z, l1.l)
	layer(l1.l, l2.w, l2.b, l2.z, l2.l)
	layer2(l2.l, l3.w, l3.b, l3.z, l3.l)
}

mat_act :: proc(a: Matrix) #no_bounds_check {
	for &d in a.data {
		d = relu(d)
	}
}

relu :: proc(x: f32) -> f32 {
    return x > 0 ? x : 0.01 * x  // leaky!!
}

relu_derive :: proc(x: f32) -> f32 {
    return x > 0 ? 1 : 0.01  // never fully zero!!
}

// last activation layer
softmax :: proc(a: Matrix) #no_bounds_check {
    // find max for numerical stability (prevents exp() overflow)
    max: f32 = a.data[0]
    for d in a.data do max = math.max(max, d)

    sum: f32
    for &d in a.data {
        d = math.exp(d - max)
        sum += d
    }
    for &d in a.data do d /= sum
}

// loss function
cross_entropy :: proc(output, target: Matrix) -> f32 #no_bounds_check {
    loss: f32
    for i in 0..<output.cols {
        loss -= target.data[i] * math.ln(output.data[i] + 1e-7)
    }
    return loss
}

argmax :: proc(a: Matrix) -> int #no_bounds_check {
    best_idx: int
    best_val := a.data[0]
    for d, i in a.data {
        if d > best_val {
            best_val = d
            best_idx = i
        }
    }
    return best_idx
}
