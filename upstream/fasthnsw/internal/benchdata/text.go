package benchdata

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// LoadTextVectors reads a simple whitespace-separated vector text file.
func LoadTextVectors(path string) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open vector file: %w", err)
	}
	defer file.Close()
	vectors, err := ParseTextVectors(file)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return vectors, nil
}

// ParseTextVectors reads one float32 vector per non-empty, non-comment line.
func ParseTextVectors(r io.Reader) ([][]float32, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), 1024*1024)

	var vectors [][]float32
	dim := 0
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if dim == 0 {
			dim = len(fields)
		}
		if len(fields) != dim {
			return nil, fmt.Errorf("line %d has dimension %d, want %d", lineNumber, len(fields), dim)
		}
		vector := make([]float32, dim)
		for i, field := range fields {
			value, err := strconv.ParseFloat(field, 32)
			if err != nil {
				return nil, fmt.Errorf("line %d field %d: %w", lineNumber, i+1, err)
			}
			vector[i] = float32(value)
		}
		vectors = append(vectors, vector)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, errors.New("vector file has no vectors")
	}
	return vectors, nil
}
