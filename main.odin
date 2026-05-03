package main

import "core:math/rand"
import "core:math"
import "core:fmt"

TRAIN: [][3]f32 = {
	{0, 0, 0},
	{0, 1, 0},
	{1, 0, 0},
	{1, 1, 1},
}

D  :: 0.001
LR :: 0.1  // learning rate

main :: proc() {
	w := mat_make(2, 1)
	mat_rand(w)
	b := mat_make(1, 1)
	mat_rand(b)

	dw := mat_make(2, 1)  // same dim as w
	db := mat_make(1, 1)  // same dim as b
	for _ in 0..<1000*100 {
		l := loss(w, b)
		// l  := loss(w1, w2, w3, w4, b1, b2, w5, w6, b3)
		for &d, idx in w.data {
			d += D
			defer d -= D
			dw.data[idx] = (loss(w, b)-l) / D
		}
		// dw1 := (loss(w1+D, w2,   w3,   w4,   b1,   b2,   w5, w6, b3) - l) / D
		for &d, idx in b.data {
			d += D
			defer d -= D
			db.data[idx] = (loss(w, b)-l) / D
		}
		
		// w1 -= LR*dw1
		for &d, idx in w.data {
			d -= dw.data[idx]*LR
		}
		for &d, idx in b.data {
			d -= db.data[idx]*LR
		}
	}

	for tr in TRAIN {
		a := mat_make(1, 2)
		a.data[0] = tr[0]
		a.data[1] = tr[1]
		o := mat_make(1, 1)
		mat_calc(a, w, b, o)
		fmt.println(tr, o.data[0])
		// fmt.println(tr, calc(tr[0], tr[1], w1, w2, w3, w4, b1, b2, w5, w6, b3))
	}
}

loss :: proc(w, b: Matrix) -> (total_cost: f32) {
	// o := mat_make(w.rows, b.cols)
	o := mat_make(1, 1)
	for tr in TRAIN {
		a := mat_make(1, 2)
		a.data[0] = tr[0]
		a.data[1] = tr[1]
		mat_calc(a, w, b, o)
		// c := calc(tr[0], tr[1], w1, w2, w3, w4, b1, b2, w5, w6, b3)
		total_cost += cost(tr[2], o.data[0])
	}
	return

	cost :: proc(a, b: f32) -> f32 {
		return (a-b)*(a-b)
	}
}

mat_calc :: proc(a, w, b, o: Matrix) {
	mat_mul(a, w, o)
	mat_add(o, b, o)
	mat_act(o)
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

calc :: proc(a1, a2, w1, w2, w3, w4, b1, b2, w5, w6, b3: f32) -> (c: f32) {
	a3 := activation(a1*w1 + a2*w3 + b1)
	a4 := activation(a1*w2 + a2*w4 + b2)
	c   = activation(a3*w5 + a4*w6 + b3)
	return
}

activation :: proc(x: f32) -> f32 {
	return 1/(1+math.exp(-x))
}
