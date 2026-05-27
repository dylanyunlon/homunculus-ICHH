package core

import "errors"

// ErrNotImplemented is reserved for public API methods that may be introduced
// before their full implementation is available.
var ErrNotImplemented = errors.New("fasthnsw: not implemented")

// ErrIndexNotBuilt is returned by Search when an index has vectors but no
// searchable graph metadata yet.
var ErrIndexNotBuilt = errors.New("fasthnsw: index is not built")
