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

main :: proc() {
	w1 := rand.float32()
	w2 := rand.float32()
	w3 := rand.float32()
	w4 := rand.float32()
	b1 := rand.float32()
	b2 := rand.float32()

	w5 := rand.float32()
	w6 := rand.float32()
	b3 := rand.float32()

	for _ in 0..<1000*100 {
		l  := loss(w1, w2, w3, w4, b1, b2, w5, w6, b3)

		dw1 := (loss(w1+D, w2,   w3,   w4,   b1,   b2,   w5, w6, b3) - l) / D
		dw2 := (loss(w1,   w2+D, w3,   w4,   b1,   b2,   w5, w6, b3) - l) / D
		dw3 := (loss(w1,   w2,   w3+D, w4,   b1,   b2,   w5, w6, b3) - l) / D
		dw4 := (loss(w1,   w2,   w3,   w4+D, b1,   b2,   w5, w6, b3) - l) / D
		db1 := (loss(w1,   w2,   w3,   w4,   b1+D, b2,   w5, w6, b3) - l) / D
		db2 := (loss(w1,   w2,   w3,   w4,   b1,   b2+D, w5, w6, b3) - l) / D

		dw5 := (loss(w1, w2, w3, w4, b1, b2, w5+D, w6,   b3  ) - l) / D
		dw6 := (loss(w1, w2, w3, w4, b1, b2, w5,   w6+D, b3  ) - l) / D
		db3 := (loss(w1, w2, w3, w4, b1, b2, w5,   w6,   b3+D) - l) / D

		w1 -= LR*dw1
		w2 -= LR*dw2
		w3 -= LR*dw3
		w4 -= LR*dw4
		b1 -= LR*db1
		b2 -= LR*db2

		w5 -= LR*dw5
		w6 -= LR*dw6
		b3 -= LR*db3
	}

	for tr in TRAIN {
		fmt.println(tr, calc(tr[0], tr[1], w1, w2, w3, w4, b1, b2, w5, w6, b3))
	}
}

loss :: proc(w1, w2, w3, w4, b1, b2, w5, w6, b3: f32) -> (total_cost: f32) {
	for tr in TRAIN {
		c := calc(tr[0], tr[1], w1, w2, w3, w4, b1, b2, w5, w6, b3)
		total_cost += cost(tr[2], c)
	}
	return

	cost :: proc(a, b: f32) -> f32 {
		return (a-b)*(a-b)
	}
}


calc :: proc(a1, a2, w1, w2, w3, w4, b1, b2, w5, w6, b3: f32) -> (c: f32) {
	a3 := activation(a1*w1 + a2*w3 + b1)
	a4 := activation(a1*w2 + a2*w4 + b2)
	c   = activation(a3*w5 + a4*w6 + b3)
	return

	activation :: proc(x: f32) -> f32 {
		return 1/(1+math.exp(-x))
	}
}

