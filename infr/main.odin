package infr

import "core:slice"
import "core:os"
import "core:fmt"
import rl "vendor:raylib"
import ai "../"

vec2 :: [2]f32

SZ :: 28
SC :: 20

main :: proc() {
	wb, err := os.read_entire_file("wb.f32", context.allocator)
	assert(err == nil)
	wbf := slice.reinterpret([]f32, wb)

	rl.SetTraceLogLevel(.WARNING)
	rl.InitWindow(800, 600, "window")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	// img: [SZ*SZ]f32
	img := ai.mat_make(1, SZ*SZ)

	l1 := ai.make_layer(28*28, 128)
	l2 := ai.make_layer(128, 64)
	l3 := ai.make_layer(64, 10)

	idx := 0
	idx += copy(l1.w.data, wbf[idx:])
	idx += copy(l1.b.data, wbf[idx:])
	idx += copy(l2.w.data, wbf[idx:])
	idx += copy(l2.b.data, wbf[idx:])
	idx += copy(l3.w.data, wbf[idx:])
	idx += copy(l3.b.data, wbf[idx:])

	fmt.println("l1.w sample:", l1.w.data[:5])
	fmt.println("l1.b sample:", l1.b.data[:5])

	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(.SPACE) {
			for &d in img.data {
				d = 0
			}
		}

		if rl.IsMouseButtonDown(.LEFT) {
			pos := rl.GetMousePosition()
			pos /= SC
			x, y := int(pos.x), int(pos.y)
			i := x+y*SZ

			img.data[i] = 1
			img.data[i-1] = 1
			img.data[i+1] = 1
			img.data[i+SZ] = 1
			img.data[i-SZ] = 1

			ai.mat_calc(img, l1, l2, l3)
		}

		rl.BeginDrawing()
		rl.ClearBackground(rl.RAYWHITE)

		for x in 0..<SZ {
			for y in 0..<SZ {
				i := x+y*SZ
				p := img.data[i]
				c := u8(p*255)
				cc := rl.Color{c,c,c,255}
				rl.DrawRectangleV({f32(x),f32(y)}*SC, SC, cc)
			}
		}

		max_l: f32 = 0
		max_l_idx := 0
		for l, l_idx in l3.l.data {
			if l > max_l {
				max_l = l
				max_l_idx = l_idx
			}
		}
		for l, l_idx in l3.l.data {
			// fmt.println(l_idx, l)
			color := l_idx == max_l_idx ? rl.GREEN : rl.BLACK
			txt := rl.TextFormat("%d %f", l_idx, l)
			rl.DrawText(txt, SC*SZ, i32(l_idx)*20, 20, color)
		}
		
		rl.EndDrawing()
	}
}
