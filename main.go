// Copyright 2016 The NetPress Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/disintegration/gift"
	"github.com/nfnt/resize"
	"github.com/pointlander/compress"
	"github.com/pointlander/neural"
)

const (
	testImage    = "images/image01.png"
	blockSize    = 4
	netWidth     = blockSize * blockSize
	hiddens      = netWidth / 2
	hiddenLayer  = 2
	quantization = 4
	scale        = 1
)

func Gray(input image.Image) *image.Gray16 {
	bounds := input.Bounds()
	output := image.NewGray16(bounds)
	width, height := bounds.Max.X, bounds.Max.Y
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := input.At(x, y).RGBA()
			output.SetGray16(x, y, color.Gray16{uint16((float64(r)+float64(g)+float64(b))/3 + .5)})
		}
	}
	return output
}

func press(input *bytes.Buffer) *bytes.Buffer {
	data, in, output := make([]byte, input.Len()), make(chan []byte, 1), &bytes.Buffer{}
	copy(data, input.Bytes())
	in <- data
	close(in)
	compress.BijectiveBurrowsWheelerCoder(in).MoveToFrontRunLengthCoder().AdaptiveCoder().Code(output)
	return output
}

func unpress(input *bytes.Buffer, size int) *bytes.Buffer {
	data, in := make([]byte, size), make(chan []byte, 1)
	in <- data
	close(in)
	compress.BijectiveBurrowsWheelerDecoder(in).MoveToFrontRunLengthDecoder().AdaptiveDecoder().Decode(input)
	return bytes.NewBuffer(data)
}

func main() {
	file, err := os.Open(testImage)
	if err != nil {
		log.Fatal(err)
	}

	info, err := file.Stat()
	if err != nil {
		log.Fatal(err)
	}
	name := info.Name()
	name = name[:strings.Index(name, ".")]

	input, _, err := image.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	width, height := input.Bounds().Max.X, input.Bounds().Max.Y
	width, height = width/scale, height/scale
	input = resize.Resize(uint(width), uint(height), input, resize.NearestNeighbor)
	width -= width % blockSize
	height -= height % blockSize
	bounds := image.Rect(0, 0, width, height)
	g := gift.New(
		gift.Crop(bounds),
	)
	cropped := image.NewGray16(bounds)
	g.Draw(cropped, input)
	input = Gray(cropped)

	size := width * height / netWidth

	file, err = os.Create(name + ".png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, input)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	file, err = os.Create(name + ".jpg")
	if err != nil {
		log.Fatal(err)
	}

	err = jpeg.Encode(file, input, &jpeg.Options{Quality: 45})
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	config := func(n *neural.Neural32) {
		n.Init(neural.WeightInitializer32FanIn, netWidth, netWidth, hiddens, netWidth, netWidth)
		mask, layer := uint8(0xFF), hiddenLayer-1
		mask <<= quantization
		hidden := n.Functions[layer].F
		n.Functions[layer].F = func(x float32) float32 {
			x = hidden(x)
			return float32(uint8(x*255)&mask) / 255
		}
	}
	codec := neural.NewNeural32(config)

	patterns, c := make([][][]float32, size), 0
	for j := 0; j < height; j += blockSize {
		for i := 0; i < width; i += blockSize {
			pixels, p := make([]float32, netWidth), 0
			for y := 0; y < blockSize; y++ {
				for x := 0; x < blockSize; x++ {
					pixel, _, _, _ := input.At(i+x, j+y).RGBA()
					pixels[p] = float32(pixel) / 0xFFFF
					p++
				}
			}
			patterns[c] = [][]float32{pixels, pixels}
			c++
		}
	}
	randomized := make([][][]float32, size)
	copy(randomized, patterns)
	source := func(iterations int) [][][]float32 {
		for i := 0; i < size; i++ {
			j := i + rand.Intn(size-i)
			randomized[i], randomized[j] = randomized[j], randomized[i]
		}
		return randomized
	}
	codec.Train(source, 10, 0.6, 0.4)
	context := codec.NewContext()

	type Stat struct {
		sum, min, max float32
	}
	stats := make([]Stat, hiddens)
	for i := range stats {
		stats[i].min = math.MaxFloat32
	}

	coded := image.NewGray16(input.Bounds())

	c = 0
	for j := 0; j < height; j += blockSize {
		for i := 0; i < width; i += blockSize {
			pattern := patterns[c]

			context.SetInput(pattern[0])
			context.Infer()
			outputs, o := context.GetOutput(), 0
			for k, act := range context.Activations[hiddenLayer][:hiddens] {
				stats[k].sum += act
				if act < stats[k].min {
					stats[k].min = act
				}
				if act > stats[k].max {
					stats[k].max = act
				}
			}

			for y := 0; y < blockSize; y++ {
				for x := 0; x < blockSize; x++ {
					coded.SetGray16(i+x, j+y, color.Gray16{uint16(outputs[o]*0xFFFF + .5)})
					o++
				}
			}

			c++
		}
	}

	for i := range stats {
		stats[i].sum /= float32(len(patterns))
	}
	fmt.Println(stats)

	file, err = os.Create(name + "_autocoded.png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, coded)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	output := make([][]uint8, hiddens)
	for i := range output {
		output[i] = make([]uint8, size)
	}

	c = 0
	for j := 0; j < height; j += blockSize {
		for i := 0; i < width; i += blockSize {
			pattern := patterns[c]

			context.SetInput(pattern[0])
			context.Infer()
			for k, act := range context.Activations[hiddenLayer][:hiddens] {
				min, max := stats[k].min, stats[k].max
				output[k][c] = uint8(255 * (act - min) / (max - min))
			}

			c++
		}
	}

	for i, set := range output {
		cwidth, cheight := width/blockSize, height/blockSize
		component, s := image.NewGray(image.Rect(0, 0, cwidth, cheight)), 0
		for y := 0; y < cheight; y++ {
			for x := 0; x < cwidth; x++ {
				component.SetGray(x, y, color.Gray{set[s]})
				set[s] >>= quantization
				s++
			}
		}
		file, err = os.Create(fmt.Sprintf("%v_%v.png", name, i))
		if err != nil {
			log.Fatal(err)
		}

		err = png.Encode(file, component)
		if err != nil {
			log.Fatal(err)
		}
		file.Close()
	}

	totalUncompressed, totalCompressed := 0, 0
	for _, set := range output {
		totalUncompressed += len(set)
		pressed := press(bytes.NewBuffer(set))
		totalCompressed += pressed.Len()
	}
	fmt.Println(totalCompressed, float64(totalCompressed)/float64(totalUncompressed))

	decoded := image.NewGray16(input.Bounds())
	config = func(n *neural.Neural32) {
		n.Init(neural.WeightInitializer32FanIn, hiddens, netWidth, netWidth)
		for l := range n.Weights {
			for i := range n.Weights[l] {
				for j := range n.Weights[l][i] {
					n.Weights[l][i][j] = codec.Weights[hiddenLayer+l][i][j]
				}
			}
		}
	}
	decoder := neural.NewNeural32(config)
	context = decoder.NewContext()

	c = 0
	pattern := make([]float32, hiddens)
	for j := 0; j < height; j += blockSize {
		for i := 0; i < width; i += blockSize {
			for k := 0; k < hiddens; k++ {
				min, max := stats[k].min, stats[k].max
				pattern[k] = float32(output[k][c]<<quantization)*(max-min)/255 + min
			}

			context.SetInput(pattern)
			context.Infer()
			outputs, o := context.GetOutput(), 0
			for y := 0; y < blockSize; y++ {
				for x := 0; x < blockSize; x++ {
					decoded.SetGray16(i+x, j+y, color.Gray16{uint16(outputs[o]*0xFFFF + .5)})
					o++
				}
			}

			c++
		}
	}

	/*g = gift.New(
		gift.Convolution(
			[]float32{
				1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
				1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
				1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
			},
			false, false, false, 0,
		),
	)
	filtered := image.NewGray16(input.Bounds())
	g.Draw(filtered, decoded)*/

	file, err = os.Create(name + "_decoded.png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, decoded)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

}
