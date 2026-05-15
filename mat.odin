#+vet explicit-allocators
package main
// note: claude-generated /shrug
import "core:math/rand"
import "core:math"

import "core:fmt"

Matrix :: struct {
	data: []f32,
	rows: int,
	cols: int,
}

mat_transpose :: proc(a: Matrix) -> Matrix {
	return {
		data = a.data,
		rows = a.cols,
		cols = a.rows,
	}
}

mat_make :: proc(rows, cols: int, allocator := context.allocator) -> Matrix {
	return {
		data = make([]f32, rows * cols, allocator),
		rows = rows,
		cols = cols,
	}
}

// mat_make_random :: proc(rows, cols: int, allocator := context.allocator) -> Matrix {
// 	c := mat_make(rows, cols, allocator)
// 	mat_rand(c)
// 	return c
// }

mat_make_random :: proc(inputs, outputs: int, allocator := context.allocator) -> Matrix {
    m := mat_make(inputs, outputs, allocator)
    scale := math.sqrt(2.0 / f32(inputs))  // He initialization for ReLU
    for &d in m.data {
        d = (rand.float32()*2 - 1) * scale
    }
    return m
}

mat_make_from :: proc(m: Matrix, allocator := context.allocator) -> Matrix {
	return mat_make(m.rows, m.cols, allocator)
}

mat_free :: proc(m: Matrix, allocator := context.allocator) {
	delete(m.data, allocator)
}

mat_get :: #force_inline proc(m: Matrix, r, c: int) -> f32 #no_bounds_check {
	return m.data[r * m.cols + c]
}

mat_set :: #force_inline proc(m: Matrix, r, c: int, val: f32) #no_bounds_check {
	m.data[r * m.cols + c] = val
}

mat_add :: #force_inline proc(a, b: Matrix, c: Matrix) #no_bounds_check {
	// assert(a.rows == b.rows && a.cols == b.cols, "mat_add: shape mismatch")
	// assert(c.rows == a.rows && c.cols == a.cols, "mat_add: output size mismatch")
	for i in 0..<len(a.data) {
		c.data[i] = a.data[i] + b.data[i]
	}
}

mat_el_mul :: #force_inline proc(a, b: Matrix, c: Matrix) #no_bounds_check {
	// assert(a.rows == b.rows && a.cols == b.cols, "mat_el_mul: shape mismatch")
	// assert(c.rows == a.rows && c.cols == a.cols, "mat_el_mul: output size mismatch")
	for i in 0..<len(a.data) {
		c.data[i] = a.data[i] * b.data[i]
	}
}

mat_sub :: #force_inline proc(a, b: Matrix, c: Matrix) #no_bounds_check {
	// assert(a.rows == b.rows && a.cols == b.cols, "mat_sub: shape mismatch")
	// assert(c.rows == a.rows && c.cols == a.cols, "mat_sub: output size mismatch")
	for i in 0..<len(a.data) {
		c.data[i] = a.data[i] - b.data[i]
	}
}

mat_offset :: #force_inline proc(a: Matrix, b: f32, c: Matrix) #no_bounds_check {
	// assert(c.rows == a.rows && c.cols == a.cols, "mat_offset: output size mismatch")
	for i in 0..<len(a.data) {
		c.data[i] = a.data[i] + b
	}
}

mat_scale :: #force_inline proc(a: Matrix, b: f32, c: Matrix) #no_bounds_check {
	// assert(c.rows == a.rows && c.cols == a.cols, "mat_scale: output size mismatch")
	for i in 0..<len(a.data) {
		c.data[i] = a.data[i] * b
	}
}

mat_neg :: #force_inline proc(a: Matrix, c: Matrix) #no_bounds_check {
	// assert(c.rows == a.rows && c.cols == a.cols, "mat_neg: output size mismatch")
	for i in 0..<len(a.data) {
		c.data[i] = -a.data[i]
	}
}

mat_copy :: #force_inline proc(a: Matrix, c: Matrix) #no_bounds_check {
	// assert(c.rows == a.rows && c.cols == a.cols, "mat_copy: output size mismatch")
	for i in 0..<len(a.data) {
		c.data[i] = a.data[i]
	}
}

mat_mul :: proc(
	a, b, c: Matrix,
	// _a := #caller_expression(a),
	// _b := #caller_expression(b),
	// _c := #caller_expression(c),
	// loc := #caller_location,
) #no_bounds_check {
	// assert_eq(a.cols, b.rows, loc=loc)
	// if a.cols != b.rows {
	// 	fmt.panicf(
	// 		"%v.cols(%d) != %v.rows(%d)\n%v=[%d,%d]\n%v=[%d,%d]",
	// 		_a, a.cols,
	// 		_b, b.rows,
	// 		_a, a.rows, a.cols,
	// 		_b, b.rows, b.cols,
	// 		loc=loc,
	// 	)
	// }
	// assert_eq(c.rows, a.rows, loc=loc)
	// // assert_eq(c.cols, b.cols, loc=loc)
	// if c.cols != b.cols {
	// 	fmt.panicf(
	// 		"%v.cols(%d) != %v.cols(%d)\n%v=[%d,%d]\n%v=[%d,%d]",
	// 		_c, c.cols,
	// 		_b, b.cols,
	// 		_c, c.rows, c.cols,
	// 		_b, b.rows, b.cols,
	// 		loc=loc,
	// 	)
	// }
	for r in 0..<a.rows {
		for c_col in 0..<b.cols {
			sum: f32 = 0
			for k in 0..<a.cols {
				sum += mat_get(a, r, k) * mat_get(b, k, c_col)
			}
			mat_set(c, r, c_col, sum)
		}
	}
}

assert_eq :: proc(a, b: $T, A := #caller_expression(a), B := #caller_expression(b), loc := #caller_location) {
	fmt.assertf(a == b, "Expected %v=%v but got %v=%v", A, a, B, b, loc=loc)
}
