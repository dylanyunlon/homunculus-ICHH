package main

import (
	"fmt"
	"os"

	"github.com/cryo-zd/fasthnsw"
)

func main() {
	cfg := fasthnsw.DefaultConfig()
	cfg.Dim = 2
	cfg.M = 4
	cfg.K0 = 4
	cfg.CandidateK = 4
	cfg.ConstructionL = 4

	idx, err := fasthnsw.New(cfg)
	if err != nil {
		panic(err)
	}
	if err := idx.Build([][]float32{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
	}); err != nil {
		panic(err)
	}

	file, err := os.CreateTemp("", "fasthnsw-*.idx")
	if err != nil {
		panic(err)
	}
	defer os.Remove(file.Name())

	if err := idx.Save(file); err != nil {
		panic(err)
	}
	if err := file.Close(); err != nil {
		panic(err)
	}

	file, err = os.Open(file.Name())
	if err != nil {
		panic(err)
	}
	defer file.Close()

	loaded, err := fasthnsw.Load(file)
	if err != nil {
		panic(err)
	}
	results, err := loaded.Search([]float32{0, 0}, 2, 4)
	if err != nil {
		panic(err)
	}
	for _, result := range results {
		fmt.Printf("id=%d distance=%.6f\n", result.ID, result.Distance)
	}
}
