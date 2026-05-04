#+vet explicit-allocators
package main
// note: claude-generated /shrug

import "core:fmt"
Matrix :: struct {
	data: []f32,
	rows: int,
	cols: int,
}

mat_make :: proc(rows, cols: int, allocator := context.allocator) -> Matrix {
	return {
		data = make([]f32, rows * cols, allocator),
		rows = rows,
		cols = cols,
	}
}

mat_make_from :: proc(m: Matrix, allocator := context.allocator) -> Matrix {
    return mat_make(m.rows, m.cols, allocator)
}

mat_free :: proc(m: Matrix, allocator := context.allocator) {
	delete(m.data, allocator)
}

mat_get :: proc(m: Matrix, r, c: int) -> f32 {
	return m.data[r * m.cols + c]
}

mat_set :: proc(m: Matrix, r, c: int, val: f32) {
	m.data[r * m.cols + c] = val
}

mat_add :: proc(a, b: Matrix, c: Matrix) {
    assert(a.rows == b.rows && a.cols == b.cols, "mat_add: shape mismatch")
    assert(c.rows == a.rows && c.cols == a.cols, "mat_add: output size mismatch")
    for i in 0..<len(a.data) {
        c.data[i] = a.data[i] + b.data[i]
    }
}

mat_mul :: proc(a, b: Matrix, c: Matrix, loc := #caller_location) {
    assert_eq(a.cols, b.rows, loc=loc)
    assert_eq(c.rows, a.rows, loc=loc)
    assert_eq(c.cols, b.cols, loc=loc)
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
