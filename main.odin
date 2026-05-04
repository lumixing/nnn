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
	dw1 := mat_make_from(w1)

	b1 := mat_make(VEC, MIDDLES); mat_rand(b1)
	db1 := mat_make_from(b1)
	l1  := mat_make_from(b1)

	w2 := mat_make(MIDDLES, OUTPUTS); mat_rand(w2)
	dw2 := mat_make_from(w2)

	b2 := mat_make(VEC, OUTPUTS); mat_rand(b2)
	db2 := mat_make_from(b2)
	l2  := mat_make_from(b2)

	for _ in 0..<1000*100 {
		l := loss(w1, b1, w2, b2, a, l1, l2)
		for &d, idx in w1.data {
			d += D; defer d -= D
			dw1.data[idx] = (loss(w1, b1, w2, b2, a, l1, l2)-l) / D
		}
		for &d, idx in b1.data {
			d += D; defer d -= D
			db1.data[idx] = (loss(w1, b1, w2, b2, a, l1, l2)-l) / D
		}
		for &d, idx in w2.data {
			d += D; defer d -= D
			dw2.data[idx] = (loss(w1, b1, w2, b2, a, l1, l2)-l) / D
		}
		for &d, idx in b2.data {
			d += D; defer d -= D
			db2.data[idx] = (loss(w1, b1, w2, b2, a, l1, l2)-l) / D
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
		mat_calc(a, w1, b1, w2, b2, l1, l2)
		fmt.println(tr, l2.data[0])
	}
}

layer :: proc(a, w, b, l: Matrix) {
	assert_eq(l.rows, b.rows); assert_eq(l.cols, b.cols)

	mat_mul(a, w, l)
	mat_add(l, b, l)
	mat_act(l)
	return
}

loss :: proc(w1, b1, w2, b2, a, l1, l2: Matrix) -> (total_cost: f32) {
	assert_eq(a.rows, 1); assert_eq(a.cols, 2)
	assert_eq(l1.rows, b1.rows); assert_eq(l1.cols, b1.cols)
	assert_eq(l2.rows, b2.rows); assert_eq(l2.cols, b2.cols)

	for tr in TRAIN {
		a.data[0] = tr[0]; a.data[1] = tr[1]
		mat_calc(a, w1, b1, w2, b2, l1, l2)
		total_cost += cost(tr[2], l2.data[0])  // safe
	}
	return

	cost :: proc(a, b: f32) -> f32 {
		return (a-b)*(a-b)
	}
}

mat_calc :: proc(a, w1, b1, w2, b2, l1, l2: Matrix) {
	assert_eq(l1.rows, b1.rows); assert_eq(l1.cols, b1.cols)
	assert_eq(l2.rows, b2.rows); assert_eq(l2.cols, b2.cols)

	layer(a, w1, b1, l1)
	layer(l1, w2, b2, l2)
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
