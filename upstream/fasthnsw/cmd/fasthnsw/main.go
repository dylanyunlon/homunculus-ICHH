package main

import (
	"fmt"
	"io"
	"os"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout io.Writer, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stderr)
		return 2
	}

	var err error
	switch args[0] {
	case "build":
		err = runBuild(args[1:], stdout)
	case "query":
		err = runQuery(args[1:], stdout)
	case "validate":
		err = runValidate(args[1:], stdout)
	case "help", "-help", "--help":
		printUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fasthnsw: unknown subcommand %q\n", args[0])
		printUsage(stderr)
		return 2
	}
	if err != nil {
		fmt.Fprintf(stderr, "fasthnsw: %v\n", err)
		return 1
	}
	return 0
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fasthnsw <build|query|validate> [flags]")
	fmt.Fprintln(w, "  build -input vectors.txt -output index.fhnsw [config flags]")
	fmt.Fprintln(w, "  query -index index.fhnsw -queries queries.txt -k 10 -ef 64")
	fmt.Fprintln(w, "  validate [-algorithm fasthnsw|hnsw] -dataset clustered|uniform|hdf5|fvecs|bvecs [dataset flags] -k 10 -ef 64")
}
