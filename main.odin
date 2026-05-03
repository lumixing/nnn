// #+vet explicit-allocators
package main

import "core:math/rand"
import "core:math"
import "core:fmt"

TRAIN: [][3]f32 = {
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

D  :: 0.001
LR :: 0.1  // learning rate

VEC :: 1

INPUTS :: 2
MIDDLES :: 2
OUTPUTS :: 1

main :: proc() {
	a := mat_make(VEC, INPUTS)

	w1 := mat_make(INPUTS, MIDDLES); mat_rand(w1)
	dw1 := mat_make(w1.rows, w1.cols)

	b1 := mat_make(VEC, MIDDLES); mat_rand(b1)
	db1 := mat_make(b1.rows, b1.cols)
	l1 := mat_make(b1.rows, b1.cols)

	w2 := mat_make(MIDDLES, OUTPUTS); mat_rand(w2)
	dw2 := mat_make(w2.rows, w2.cols)

	b2 := mat_make(VEC, OUTPUTS); mat_rand(b2)
	db2 := mat_make(b2.rows, b2.cols)
	l2 := mat_make(b2.rows, b2.cols)

	o := mat_make(VEC, VEC)

	for _ in 0..<1000*100 {
		l := loss(w1, b1, w2, b2, o, a, l1, l2)
		for &d, idx in w1.data {
			d += D; defer d -= D
			dw1.data[idx] = (loss(w1, b1, w2, b2, o, a, l1, l2)-l) / D
		}
		for &d, idx in b1.data {
			d += D; defer d -= D
			db1.data[idx] = (loss(w1, b1, w2, b2, o, a, l1, l2)-l) / D
		}
		for &d, idx in w2.data {
			d += D; defer d -= D
			dw2.data[idx] = (loss(w1, b1, w2, b2, o, a, l1, l2)-l) / D
		}
		for &d, idx in b2.data {
			d += D; defer d -= D
			db2.data[idx] = (loss(w1, b1, w2, b2, o, a, l1, l2)-l) / D
		}
		for &d, idx in w1.data {
			d -= dw1.data[idx]*LR
		}
		for &d, idx in b1.data {
			d -= db1.data[idx]*LR
		}
		for &d, idx in w2.data {
			d -= dw2.data[idx]*LR
		}
		for &d, idx in b2.data {
			d -= db2.data[idx]*LR
		}
	}

	for tr in TRAIN {
		a := mat_make(1, 2)
		a.data[0] = tr[0]
		a.data[1] = tr[1]
		o := mat_make(1, 1)
		mat_calc(a, w1, b1, w2, b2, o, l1, l2)
		fmt.println(tr, o.data[0])
		// fmt.println(tr, calc(tr[0], tr[1], w1, w2, w3, w4, b1, b2, w5, w6, b3))
	}
}

layer :: proc(a, w, b, o: Matrix) {
	// o = mat_make(b.rows, b.cols)
	assert_eq(o.rows, b.rows); assert_eq(o.cols, b.cols)
	// fmt.println(a, w, b, o)
	mat_mul(a, w, o)
	mat_add(o, b, o)
	mat_act(o)
	return
}

loss :: proc(w1, b1, w2, b2, o, a, l1, l2: Matrix) -> (total_cost: f32) {
	// o := mat_make(w.rows, b.cols)
	// o := mat_make(1, 1)
	assert_eq(o.rows, 1); assert_eq(o.cols, 1)
	assert_eq(a.rows, 1); assert_eq(a.cols, 2)
	assert_eq(l1.rows, b1.rows); assert_eq(l1.cols, b1.cols)
	assert_eq(l2.rows, b2.rows); assert_eq(l2.cols, b2.cols)
	for tr in TRAIN {
		// a := mat_make(1, 2)
		a.data[0] = tr[0]; a.data[1] = tr[1]
		mat_calc(a, w1, b1, w2, b2, o, l1, l2)
		// c := calc(tr[0], tr[1], w1, w2, w3, w4, b1, b2, w5, w6, b3)
		total_cost += cost(tr[2], o.data[0])  // safe
	}
	return

	cost :: proc(a, b: f32) -> f32 {
		return (a-b)*(a-b)
	}
}

mat_calc :: proc(a, w1, b1, w2, b2, o, l1, l2: Matrix) {
	// mat_mul(a, w1, o)
	// mat_add(o, b1, o)
	// mat_act(o)
	// l1 := mat_make(b1.rows, b1.cols)
	assert_eq(l1.rows, b1.rows); assert_eq(l1.cols, b1.cols)
	layer(a, w1, b1, l1)
	// l2 := mat_make(b2.rows, b2.cols)
	assert_eq(l2.rows, b2.rows); assert_eq(l2.cols, b2.cols)
	layer(l1, w2, b2, l2)
	assert_eq(o.rows, l2.rows); assert_eq(o.cols, l2.cols)
	copy(o.data, l2.data)
}

mat_act :: proc(a: Matrix) {
	for &d in a.data {
		d = activation(d)
	}
}

mat_rand :: proc(a: Matrix) {
	for &d in a.data {
		d = rand.float32()
	}
}

activation :: proc(x: f32) -> f32 {
	return 1/(1+math.exp(-x))
}
