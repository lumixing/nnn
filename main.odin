package main

import "core:fmt"

TRAIN: [][2]f32 = {
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
}

main :: proc() {
	w: f32 = 0.5
	d: f32 = 0.0001

	for _ in 0..<1000*1000 {
		l := loss(w)
		nl := loss(w+d)
		if nl <l {
			w  += d
		} else {
			w -= d
		}
	}

	fmt.println(w, loss(w))
}

loss :: proc(w: f32) -> (total_cost: f32) {
	for tr in TRAIN {
		c := calc(tr[0], w)
		total_cost += cost(tr[1], c)
	}
	return
}

cost :: proc(a, b: f32) -> f32 {
	return (a-b)*(a-b)
}

calc :: proc(a, w: f32) -> f32 {
	return a*w
}
