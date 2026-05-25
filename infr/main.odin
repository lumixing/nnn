package infr

import "core:math"
import "core:mem"
import "core:slice"
import "core:os"
import "core:fmt"
import rl "vendor:raylib"
import ai "../"

vec2 :: [2]f32
vec3 :: [3]f32

SZ :: 28
SC :: 20

main :: proc() {
	wb, err := os.read_entire_file("wb60.f32", context.allocator)
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

	cam: rl.Camera3D = {
		position = 8,
		fovy = 45,
		up = {0,1,0},
	}

	if d, err := os.read_entire_file("camera.mem", context.allocator); err == nil {
		// cam = mem.reinterpret_copy(rl.Camera3D, raw_data(d))
		cam = mem.slice_data_cast([]rl.Camera3D, d)[0]
	}
	defer {
		_ = os.write_entire_file("camera.mem", mem.ptr_to_bytes(&cam))
	}

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

		dt := rl.GetFrameTime()
		mdt := rl.GetMouseDelta()
		rl.UpdateCameraPro(&cam, 0, 0, -rl.GetMouseWheelMove())
		if rl.IsMouseButtonDown(.RIGHT) {
			rl.CameraPitch(&cam, -mdt.y*dt, true, true, false)
			rl.CameraYaw(&cam, -mdt.x*dt, true)
		}
		if rl.IsMouseButtonDown(.MIDDLE) {
			rl.CameraMoveRight(&cam, -mdt.x*dt, false)
			rl.CameraMoveUp(&cam, mdt.y*dt)
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

		rl.BeginMode3D(cam)

		for v, i in img.data {//1x784->28*28
			x, y := i % 28, i / 28
			pos := vec3{f32(x),-f32(y),0}*1.2
			pos += {-1.2,1.2,0}*28/2
			cube(pos, scientific_color(v))
			for w, j in l1.w.data {//784*128
				
			}
		}
		for v, i in l1.l.data {
			x, y := i % 12, i / 12
			pos := vec3{f32(x),-f32(y),8}*1.2
			pos += {-1.2,1.2,0}*12/2
			cube(pos, scientific_color(v))
		}
		for v, i in l2.l.data {
			x, y := i % 8, i / 8
			pos := vec3{f32(x),-f32(y),16}*1.2
			pos += {-1.2,1.2,0}*8/2
			cube(pos, scientific_color(v))

		}
		for v, i in l3.l.data {
			pos := vec3{f32(i),0,24}*1.2
			pos += {-1.2,0,0}*8/2
			cube(pos, scientific_color(v))
		}

		rl.EndMode3D()

		rl.DrawFPS(0, 0)
		
		rl.EndDrawing()
	}
}

scientific_color :: proc(t: f32) -> rl.Color {
    hue := math.lerp(f32(270), f32(60), clamp(t, 0, 1))
    return rl.ColorFromHSV(hue, 1.0, 1.0)
}

cube :: proc(pos: vec3, col: rl.Color) {
	// rl.DrawCubeV(pos, 1, rl.ColorAlpha(col, 0.5))
	rl.DrawCubeV(pos, 1, col)
	rl.DrawCubeWiresV(pos, 1, {0,0,0,150})
}
