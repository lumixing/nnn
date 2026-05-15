package mnist

import "vendor:stb/image"
import "core:os"
import "core:fmt"

main :: proc() {
	imgdata, err := os.read_entire_file("train-images.idx3-ubyte", context.allocator)
	assert(err == nil)
	lbldata, err2 := os.read_entire_file("train-labels.idx1-ubyte", context.allocator)
	assert(err2 == nil)

	image.write_png("out.png", 28, 28, 1, raw_data(imgdata[16+28*28*1719:][:28*28]), 28)
	fmt.println(lbldata[8+1719])
}
