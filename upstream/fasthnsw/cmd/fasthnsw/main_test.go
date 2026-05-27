package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildAndQueryCommands(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1 0\n0 1\n")
	queriesPath := writeTestFile(t, dir, "queries.txt", "0 0\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
		"-dim", "2",
		"-m", "2",
		"-k0", "2",
		"-candidate-k", "2",
		"-construction-l", "2",
		"-iterations", "1",
		"-workers", "1",
		"-candidate-recall", "0.9",
		"-candidate-controls", "2",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("build exit code = %d, stderr = %s", code, stderr.String())
	}
	if _, err := os.Stat(indexPath); err != nil {
		t.Fatalf("stat built index: %v", err)
	}
	if !strings.Contains(stdout.String(), "built index vectors=3 dim=2") {
		t.Fatalf("build stdout = %q, want summary", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = run([]string{
		"query",
		"-index", indexPath,
		"-queries", queriesPath,
		"-k", "1",
		"-ef", "2",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("query exit code = %d, stderr = %s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{"query=0", "rank=0", "id=0", "distance=0.000000"} {
		if !strings.Contains(got, want) {
			t.Fatalf("query stdout = %q, want %q", got, want)
		}
	}
}

func TestRunRejectsUnknownSubcommand(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{"missing"}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("run returned success for unknown subcommand")
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Fatalf("stderr = %q, want unknown subcommand", stderr.String())
	}
}

func TestBuildRejectsMissingRequiredFlags(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{"build"}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("build returned success without required flags")
	}
	if !strings.Contains(stderr.String(), "requires -input and -output") {
		t.Fatalf("stderr = %q, want required flag error", stderr.String())
	}
}

func TestBuildRejectsInconsistentVectorDimensions(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
	}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("build returned success for inconsistent vectors")
	}
	if !strings.Contains(stderr.String(), "dimension") {
		t.Fatalf("stderr = %q, want dimension error", stderr.String())
	}
}

func TestBuildRejectsInvalidMetric(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1 0\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
		"-metric", "dot",
	}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("build returned success for invalid metric")
	}
	if !strings.Contains(stderr.String(), "unsupported metric") {
		t.Fatalf("stderr = %q, want metric error", stderr.String())
	}
}

func TestQueryRejectsWrongDimension(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1 0\n0 1\n")
	queriesPath := writeTestFile(t, dir, "queries.txt", "0 0 0\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
		"-dim", "2",
		"-m", "2",
		"-k0", "2",
		"-candidate-k", "2",
		"-construction-l", "2",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("build exit code = %d, stderr = %s", code, stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = run([]string{
		"query",
		"-index", indexPath,
		"-queries", queriesPath,
		"-k", "1",
		"-ef", "2",
	}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("query returned success for wrong query dimension")
	}
	if !strings.Contains(stderr.String(), "query vector has dimension") {
		t.Fatalf("stderr = %q, want query dimension error", stderr.String())
	}
}

func TestValidateClusteredCommand(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"validate",
		"-dataset", "clustered",
		"-vectors", "40",
		"-query-count", "8",
		"-dim", "4",
		"-clusters", "4",
		"-k", "3",
		"-ef", "8",
		"-m", "4",
		"-k0", "8",
		"-candidate-k", "8",
		"-construction-l", "12",
		"-iterations", "1",
		"-workers", "1",
		"-candidate-recall", "0.8",
		"-candidate-controls", "8",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("validate exit code = %d, stderr = %s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{"algorithm=fasthnsw", "dataset=clustered", "vectors=40", "queries=8", "qps=", "recall_at_3="} {
		if !strings.Contains(got, want) {
			t.Fatalf("validate stdout = %q, want %q", got, want)
		}
	}
}

func TestValidateStandardHNSWCommand(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"validate",
		"-algorithm", "hnsw",
		"-dataset", "clustered",
		"-vectors", "40",
		"-query-count", "8",
		"-dim", "4",
		"-clusters", "4",
		"-k", "3",
		"-ef", "8",
		"-m", "4",
		"-construction-l", "12",
		"-workers", "1",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("validate hnsw exit code = %d, stderr = %s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{"algorithm=hnsw", "dataset=clustered", "vectors=40", "queries=8", "qps=", "recall_at_3="} {
		if !strings.Contains(got, want) {
			t.Fatalf("validate hnsw stdout = %q, want %q", got, want)
		}
	}
}

func TestValidateFVECSCommand(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.fvecs")
	queries := filepath.Join(dir, "query.fvecs")
	truth := filepath.Join(dir, "truth.ivecs")
	writeCLIFVECSFile(t, base, [][]float32{{0, 0}, {1, 0}, {0, 1}})
	writeCLIFVECSFile(t, queries, [][]float32{{0, 0}})
	writeCLIIVECSFile(t, truth, [][]int{{0, 1}})

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"validate",
		"-dataset", "fvecs",
		"-base", base,
		"-queries", queries,
		"-truth", truth,
		"-dim", "2",
		"-k", "1",
		"-ef", "2",
		"-m", "2",
		"-k0", "2",
		"-candidate-k", "2",
		"-construction-l", "2",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("validate exit code = %d, stderr = %s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{"dataset=fvecs", "vectors=3", "queries=1", "recall_at_1=1.000000"} {
		if !strings.Contains(got, want) {
			t.Fatalf("validate stdout = %q, want %q", got, want)
		}
	}
}

func TestValidateRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "unknown dataset", args: []string{"validate", "-dataset", "unknown"}, want: "unsupported validation dataset"},
		{name: "bad k", args: []string{"validate", "-dataset", "clustered", "-k", "0"}, want: "positive -k"},
		{name: "bad ef", args: []string{"validate", "-dataset", "clustered", "-k", "4", "-ef", "3"}, want: "greater than or equal"},
		{name: "missing hdf5 input", args: []string{"validate", "-dataset", "hdf5"}, want: "requires -input"},
		{name: "invalid metric", args: []string{"validate", "-metric", "dot"}, want: "unsupported metric"},
		{name: "invalid algorithm", args: []string{"validate", "-algorithm", "flat"}, want: "unsupported validation algorithm"},
		{name: "invalid workers", args: []string{"validate", "-dataset", "clustered", "-workers", "-1"}, want: "Workers"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stdout bytes.Buffer
			var stderr bytes.Buffer
			code := run(tt.args, &stdout, &stderr)
			if code == 0 {
				t.Fatal("validate returned success")
			}
			if !strings.Contains(stderr.String(), tt.want) {
				t.Fatalf("stderr = %q, want %q", stderr.String(), tt.want)
			}
		})
	}
}

func writeTestFile(t *testing.T, dir string, name string, content string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write test file: %v", err)
	}
	return path
}

func writeCLIFVECSFile(t *testing.T, path string, vectors [][]float32) {
	t.Helper()
	var buf bytes.Buffer
	for _, vector := range vectors {
		mustWriteCLI(t, binary.Write(&buf, binary.LittleEndian, int32(len(vector))))
		for _, value := range vector {
			mustWriteCLI(t, binary.Write(&buf, binary.LittleEndian, value))
		}
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write fvecs fixture: %v", err)
	}
}

func writeCLIIVECSFile(t *testing.T, path string, rows [][]int) {
	t.Helper()
	var buf bytes.Buffer
	for _, row := range rows {
		mustWriteCLI(t, binary.Write(&buf, binary.LittleEndian, int32(len(row))))
		for _, value := range row {
			mustWriteCLI(t, binary.Write(&buf, binary.LittleEndian, int32(value)))
		}
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write ivecs fixture: %v", err)
	}
}

func mustWriteCLI(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("write CLI fixture: %v", err)
	}
}
