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

Layer :: struct {
	w, b:   Matrix,
	z, l:   Matrix,
	dw, db: Matrix,
	dz, dl: Matrix,
}

make_layer :: proc(inputs, outputs: int) -> Layer {
	return {
		w  = mat_make_random(inputs, outputs),
		b  = mat_make_random(VEC, outputs),
		z  = mat_make(VEC, outputs),
		l  = mat_make(VEC, outputs),
		dw = mat_make(inputs, outputs),
		db = mat_make(VEC, outputs),
		dz = mat_make(VEC, outputs),
		dl = mat_make(VEC, outputs),
	}
}

main :: proc() {
	a := mat_make(VEC, INPUTS)

	l1 := make_layer(2, 2)
	l2 := make_layer(2, 1)

	target := mat_make_from(l2.l)
	input := mat_make(VEC, INPUTS)

	for _ in 0..<1000*100 {
		for tr, tr_idx in TRAIN {
			input.data[0] = tr[0]
			input.data[1] = tr[1]
			mat_calc(input, l1, l2)
			
			// ∂cost/∂l2 = 2(l2 - target)
			target.data[0] = tr[2]
			mat_sub(l2.l, target, l2.dl)
			mat_scale(l2.dl, 2, l2.dl)
			// ∂cost/∂z2 = ∂cost/∂l2 ⊙ (l2 ⊙ (1 - l2))
			mat_neg(l2.l, l2.z)
			mat_offset(l2.z, 1, l2.z)
			mat_el_mul(l2.z, l2.l, l2.z)
			mat_el_mul(l2.dl, l2.z, l2.dz)
			// ∂cost/∂w2 = a1ᵀ · ∂cost/∂z2
			mat_mul(mat_transpose(l1.l), l2.dz, l2.dw)
			// ∂cost/∂b2 = ∂cost/∂z2
			mat_copy(l2.dz, l2.db)

			// ∂cost/∂a1 = ∂cost/∂z2 · w2ᵀ
			mat_mul(l2.dz, mat_transpose(l2.w), l1.dl)
			// ∂cost/∂z1 = ∂cost/∂a1 ⊙ (a1 ⊙ (1 - a1))
			mat_neg(l1.l, l1.z)
			mat_offset(l1.z, 1, l1.z)
			mat_el_mul(l1.z, l1.l, l1.z)
			mat_el_mul(l1.dl, l1.z, l1.dz)
			// ∂cost/∂w1 = aᵀ · ∂cost/∂z1
			mat_mul(mat_transpose(input), l1.dz, l1.dw)
			// ∂cost/∂b1 = ∂cost/∂z1
			mat_copy(l1.dz, l1.db)

			mat_scale(l2.dw, LR, l2.dw)
			mat_scale(l2.db, LR, l2.db)
			mat_scale(l1.dw, LR, l1.dw)
			mat_scale(l1.db, LR, l1.db)

			mat_sub(l2.w, l2.dw, l2.w)
			mat_sub(l2.b, l2.db, l2.b)
			mat_sub(l1.w, l1.dw, l1.w)
			mat_sub(l1.b, l1.db, l1.b)
		}
	}

	for tr in TRAIN {
		a := mat_make(1, 2)
		a.data[0] = tr[0]
		a.data[1] = tr[1]
		mat_calc(a, l1, l2)
		fmt.println(tr, l2.l.data[0])
	}
}

layer :: proc(a, w, b, l: Matrix) {
	assert_eq(l.rows, b.rows); assert_eq(l.cols, b.cols)

	mat_mul(a, w, l)
	mat_add(l, b, l)
	mat_act(l)
	return
}

loss :: proc(l1, l2: Layer, a: Matrix) -> (total_cost: f32) {
	assert_eq(a.rows, 1); assert_eq(a.cols, 2)
	assert_eq(l1.l.rows, l1.b.rows); assert_eq(l1.l.cols, l1.b.cols)
	assert_eq(l2.l.rows, l2.b.rows); assert_eq(l2.l.cols, l2.b.cols)

	for tr in TRAIN {
		a.data[0] = tr[0]; a.data[1] = tr[1]
		mat_calc(a, l1, l2)
		total_cost += cost(tr[2], l2.l.data[0])  // safe
	}
	return
}

cost :: proc(a, b: f32) -> f32 {
	return (a-b)*(a-b)
}

cost_derive :: proc(a, b: f32) -> f32 {
	return 2*(a-b)
}

mat_calc :: proc(a: Matrix, l1, l2: Layer) {
	assert_eq(l1.l.rows, l1.b.rows); assert_eq(l1.l.cols, l1.b.cols)
	assert_eq(l2.l.rows, l2.b.rows); assert_eq(l2.l.cols, l2.b.cols)

	layer(a, l1.w, l1.b, l1.l)
	layer(l1.l, l2.w, l2.b, l2.l)
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

activation_derive :: proc(x: f32) -> f32 {
	return activation(x)*(1-activation(x))
}
