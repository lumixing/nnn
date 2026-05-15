package sim

import "core:math/rand"
import ai "../"
import rl "vendor:raylib"

vec2 :: [2]f32
COLOR_AGENT :: rl.Color{0,0,255,255}
COLOR_MARK  :: rl.Color{255,0,255,255}
VEC :: 1
INPUTS :: 2+4*2  //ag.pos,4*mk.pos
OUTPUTS :: 2  //ag.vel

main :: proc() {
	l1 := make_layer(INPUTS, 16)
	l2 := make_layer(16, 8)
	l3 := make_layer(8, OUTPUTS)

	rl.SetTraceLogLevel(.WARNING)
	rl.InitWindow(800, 600, "simulation o.o")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	points: int
	ag: Agent = {{{400,300},16}}
	mks: [dynamic; 4]Mark
	for _ in 0..<cap(mks) {
		append(&mks, Mark{
			pos = {rand.float32()*800, rand.float32()*600},
			size = 16,
		})
	}

	input := ai.mat_make(VEC, INPUTS)

	for !rl.WindowShouldClose() {
		// input.data[0] = 

		// ag.pos = rl.GetMousePosition()
		if rl.IsKeyPressed(.SPACE) {
			calc_layer(&l1, input)
			calc_layer(&l2, l1.l)
			calc_layer(&l3, l2.l)

		}

		for mk, mk_idx in mks {
			if rl.CheckCollisionRecs(body_rect(ag.body), body_rect(mk.body)) {
				unordered_remove(&mks, mk_idx)
			}
		}

		rl.BeginDrawing()
		rl.ClearBackground(rl.RAYWHITE)

		for &mk in mks {
			draw_mark(&mk)
		}
		draw_agent(&ag)

		rl.DrawText(rl.TextFormat("points %d", points), 0, 0, 20, rl.BLACK)

		rl.EndDrawing()
	}
}

Layer :: struct {
	w, b, l: ai.Matrix,
}

make_layer :: proc(inputs, outputs: int) -> Layer {
	return {
		w = ai.mat_make_random(inputs, outputs),
		b = ai.mat_make(VEC, OUTPUTS),
		l = ai.mat_make(VEC, OUTPUTS),
	}
}

calc_layer :: proc(l: ^Layer, input: ai.Matrix) {
	ai.mat_mul(input, l.w, l.l)
	ai.mat_add(l.l, l.b, l.l)
}

Body :: struct {
	pos: vec2,
	size: vec2,
}

body_rect :: proc(bd: Body) -> rl.Rectangle {
	return {bd.pos.x, bd.pos.y, bd.size.x, bd.size.y}
}

Agent :: struct {
	using body: Body,  // literal oop
}

draw_agent :: proc(ag: ^Agent) {
	rl.DrawRectangleV(ag.pos, ag.size, COLOR_AGENT)
}

Mark :: struct {
	using body: Body,
}

draw_mark :: proc(mk: ^Mark) {
	rl.DrawRectangleV(mk.pos, mk.size, COLOR_MARK)
}
