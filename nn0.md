# training the simplest ai model

You just received the following training set:

```odin
TRAIN: [][3]f32 = {
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
}
```

You know that `TRAIN[*][0]` is the input and that `TRAIN[*][1]` is the expected output, so when you receive `TRAIN[2][0] = 3` you should be expected to reply with `6 = TRAIN[2][1]`.

The way you reply is with this formula:

```odin
calc :: proc(x, w: f32) -> f32 {
	return x * w
}
```

where `x` is the input and `w` is your own adjustible *weight/paramater*.

Your job is to find the best weight value that matches the training data.

You start from a random weight value, lets say `0.4`. Lets test the case `{2, 4}`:

```odin
main :: proc() {
	w: f32 = 0.4
	our_calc := calc(TRAIN[1][0], w)
	expected_calc := TRAIN[1][1]
	fmt.println(expected_calc, our_calc)
}
```

```
4 0.8
```

Yeah we expected `4` but calculated `0.8` with our currect weight, pretty off. 
