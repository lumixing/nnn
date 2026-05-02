package main

import "core:math/rand"
import "core:math"
import "core:fmt"

TRAIN: [][3]f32 = {
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 1},
}

main :: proc() {
	w1 := rand.float32()
	w2 := rand.float32()
	b := rand.float32()
	d: f32 = 0.001

	for _ in 0..<1000*100 {
		l  := loss(w1, w2, b)
		d1 := (loss(w1+d, w2, b) - l) / d
		d2 := (loss(w1, w2+d, b) - l) / d
		db := (loss(w1, w2, b+d) - l) / d
		w1 -= 0.1 * d1
		w2 -= 0.1 * d2
		b  -= 0.1 * db
	}

	fmt.println(w1, w2, b, loss(w1, w2, b))
	for tr in TRAIN {
		fmt.println(tr, calc(tr[0], tr[1], w1, w2, b))
	}
}

loss :: proc(w1, w2, b: f32) -> (total_cost: f32) {
	for tr in TRAIN {
		c := calc(tr[0], tr[1], w1, w2, b)
		total_cost += cost(tr[2], c)
	}
	return
}

cost :: proc(a, b: f32) -> f32 {
	return (a-b)*(a-b)
}

calc :: proc(a1, a2, w1, w2, b: f32) -> (c: f32) {
	c = a1*w1 + a2*w2 + b
	c = activation(c)
	return
}

activation :: proc(x: f32) -> f32 {
	return 1/(1+math.exp(-x))
}
