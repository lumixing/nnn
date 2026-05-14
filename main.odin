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
	sdz1 := mat_make_from(l1)
	da1 := mat_make_from(l1)
	dz1 := mat_make_from(da1)

	w2 := mat_make(MIDDLES, OUTPUTS); mat_rand(w2)
	dw2 := mat_make_from(w2)
	b2 := mat_make(VEC, OUTPUTS); mat_rand(b2)
	db2 := mat_make_from(b2)
	l2  := mat_make_from(b2)
	sdz2 := mat_make_from(l2)
	da2 := mat_make_from(l2)
	dz2 := mat_make_from(da2)

	target := mat_make_from(l2)
	input := mat_make(VEC, INPUTS)

	for _ in 0..<1000*100 {
		for tr, tr_idx in TRAIN {
			input.data[0] = tr[0]
			input.data[1] = tr[1]
			mat_calc(input, w1, b1, w2, b2, l1, l2)
			
			// ∂cost/∂a2 = 2(a2 - target)
			target.data[0] = tr[2]
			mat_sub(l2, target, da2)
			mat_scale(da2, 2, da2)
			// ∂cost/∂z2 = ∂cost/∂a2 ⊙ (a2 ⊙ (1 - a2))
			mat_neg(l2, sdz2)
			mat_offset(sdz2, 1, sdz2)
			mat_el_mul(sdz2, l2, sdz2)
			mat_el_mul(da2, sdz2, dz2)
			// ∂cost/∂w2 = a1ᵀ · ∂cost/∂z2
			mat_mul(mat_transpose(l1), dz2, dw2)
			// ∂cost/∂b2 = ∂cost/∂z2
			mat_copy(dz2, db2)

			// ∂cost/∂a1 = ∂cost/∂z2 · w2ᵀ
			mat_mul(dz2, mat_transpose(w2), da1)
			// ∂cost/∂z1 = ∂cost/∂a1 ⊙ (a1 ⊙ (1 - a1))
			mat_neg(l1, sdz1)
			mat_offset(sdz1, 1, sdz1)
			mat_el_mul(sdz1, l1, sdz1)
			mat_el_mul(da1, sdz1, dz1)
			// ∂cost/∂w1 = aᵀ · ∂cost/∂z1
			mat_mul(mat_transpose(input), dz1, dw1)
			// ∂cost/∂b1 = ∂cost/∂z1
			mat_copy(dz1, db1)

			mat_scale(dw2, LR, dw2)
			mat_scale(db2, LR, db2)
			mat_scale(dw1, LR, dw1)
			mat_scale(db1, LR, db1)

			mat_sub(w2, dw2, w2)
			mat_sub(b2, db2, b2)
			mat_sub(w1, dw1, w1)
			mat_sub(b1, db1, b1)
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
}

cost :: proc(a, b: f32) -> f32 {
	return (a-b)*(a-b)
}

cost_derive :: proc(a, b: f32) -> f32 {
	return 2*(a-b)
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

activation_derive :: proc(x: f32) -> f32 {
	return activation(x)*(1-activation(x))
}
