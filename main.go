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
	"github.com/pointlander/gobrain"
)

const (
	testImage    = "images/image01.png"
	blockSize    = 4
	netWidth     = blockSize * blockSize
	hiddens      = netWidth / 4
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

	net := &gobrain.FeedForward32{}
	net.Init(netWidth, hiddens, netWidth)

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
	randomPatterns := make([][][]float32, size)
	copy(randomPatterns, patterns)
	for i := 0; i < size; i++ {
		j := i + int(rand.Float64()*float64(size-i))
		randomPatterns[i], randomPatterns[j] = randomPatterns[j], randomPatterns[i]
	}
	//net.TrainAutoEncoder(randomPatterns, 10, 0.1, 0.6, 0.4, false)
	net.TrainQuant(randomPatterns, 10, 0.6, 0.4, false, quantization)

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

			outputs, o := net.Update(pattern[0]), 0
			for k, act := range net.HiddenActivations[:hiddens] {
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

			net.Update(pattern[0])
			for k, act := range net.HiddenActivations[:hiddens] {
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

	c = 0
	pattern := make([]float32, hiddens)
	for j := 0; j < height; j += blockSize {
		for i := 0; i < width; i += blockSize {
			for k := 0; k < hiddens; k++ {
				min, max := stats[k].min, stats[k].max
				pattern[k] = float32(output[k][c]<<quantization)*(max-min)/255 + min
			}

			outputs, o := net.HalfUpdate(pattern), 0
			for y := 0; y < blockSize; y++ {
				for x := 0; x < blockSize; x++ {
					decoded.SetGray16(i+x, j+y, color.Gray16{uint16(outputs[o]*0xFFFF + .5)})
					o++
				}
			}

			c++
		}
	}

	g = gift.New(
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
	g.Draw(filtered, decoded)

	file, err = os.Create(name + "_decoded.png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, filtered)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

}
