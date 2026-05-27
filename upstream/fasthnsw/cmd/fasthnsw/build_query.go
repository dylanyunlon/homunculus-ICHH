package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/cryo-zd/fasthnsw"
	"github.com/cryo-zd/fasthnsw/internal/benchdata"
)

func runBuild(args []string, stdout io.Writer) error {
	cfgFlags := defaultConfigFlags()
	fs := newFlagSet("build")
	inputPath := fs.String("input", "", "input text vector file")
	outputPath := fs.String("output", "", "output index file")
	addConfigFlags(fs, &cfgFlags)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("build received unexpected positional arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *inputPath == "" || *outputPath == "" {
		return errors.New("build requires -input and -output")
	}

	cfg, err := cfgFlags.config()
	if err != nil {
		return err
	}
	vectors, err := benchdata.LoadTextVectors(*inputPath)
	if err != nil {
		return err
	}
	idx, err := fasthnsw.New(cfg)
	if err != nil {
		return err
	}
	if err := idx.Build(vectors); err != nil {
		return err
	}

	file, err := os.Create(*outputPath)
	if err != nil {
		return fmt.Errorf("create index file: %w", err)
	}
	if err := idx.Save(file); err != nil {
		if closeErr := file.Close(); closeErr != nil {
			return errors.Join(err, fmt.Errorf("close index file: %w", closeErr))
		}
		return err
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close index file: %w", err)
	}

	fmt.Fprintf(stdout, "built index vectors=%d dim=%d output=%s\n", len(vectors), len(vectors[0]), *outputPath)
	return nil
}

func runQuery(args []string, stdout io.Writer) error {
	fs := newFlagSet("query")
	indexPath := fs.String("index", "", "input index file")
	queriesPath := fs.String("queries", "", "input text query vector file")
	k := fs.Int("k", 10, "number of nearest neighbors")
	efSearch := fs.Int("ef", 64, "HNSW search width")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("query received unexpected positional arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *indexPath == "" || *queriesPath == "" {
		return errors.New("query requires -index and -queries")
	}

	file, err := os.Open(*indexPath)
	if err != nil {
		return fmt.Errorf("open index file: %w", err)
	}
	defer file.Close()
	idx, err := fasthnsw.Load(file)
	if err != nil {
		return err
	}

	queries, err := benchdata.LoadTextVectors(*queriesPath)
	if err != nil {
		return err
	}
	for queryID, query := range queries {
		results, err := idx.Search(query, *k, *efSearch)
		if err != nil {
			return fmt.Errorf("query %d: %w", queryID, err)
		}
		for rank, result := range results {
			fmt.Fprintf(stdout, "query=%d rank=%d id=%d distance=%.6f\n", queryID, rank, result.ID, result.Distance)
		}
	}
	return nil
}
