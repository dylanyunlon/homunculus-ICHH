package core

// search runs the standard HNSW query procedure with a query already prepared
// for the index metric. Each upper layer calls SEARCH-LAYER with ef=1, then
// layer 0 calls SEARCH-LAYER with the caller-provided efSearch and returns the
// best k results.
func (idx *Index) search(query []float32, k int, efSearch int) ([]Result, error) {
	entry := idx.entryPoint
	var scratch graphSearchScratch
	scratch.reserve(idx.count, efSearch)
	for layer := idx.maxLayer; layer > 0; layer-- {
		if !idx.searchLayerInto(layer, entry, query, 1, &scratch) {
			return nil, ErrIndexNotBuilt
		}
		entry = scratch.results[0].ID
	}

	results := idx.searchLayerTopK(0, entry, query, efSearch, k, &scratch)
	if len(results) == 0 {
		return nil, ErrIndexNotBuilt
	}
	return results, nil
}

// searchLayer implements the HNSW SEARCH-LAYER routine.
//
// The candidate set C is a nearest-first min-heap because the algorithm must
// expand the closest unexpanded candidate next. The result set W is a
// farthest-first max-heap because the algorithm repeatedly compares against,
// and evicts, the current worst retained result. Using a max-heap for C would
// expand the farthest candidate first and would not match HNSW search.
func (idx *Index) searchLayerInto(layer int, entry int, query []float32, ef int, scratch *graphSearchScratch) bool {
	return graphSearchLayerInto(idx.layers[layer], idx.vectors, idx.dim, idx.cfg.Metric, entry, query, ef, scratch)
}

func (idx *Index) searchLayerTopK(layer int, entry int, query []float32, ef int, k int, scratch *graphSearchScratch) []Result {
	return graphSearchLayerTopK(idx.layers[layer], idx.vectors, idx.dim, idx.cfg.Metric, entry, query, ef, k, scratch)
}
