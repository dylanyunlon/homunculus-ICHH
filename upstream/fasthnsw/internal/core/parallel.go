package core

import "sync"

// effectiveWorkerCount clamps a requested worker count to the amount of
// node-local work available. Non-positive values are treated as sequential
// inside hot-path helpers because public config normalization is the boundary
// that expands zero to the package default.
func effectiveWorkerCount(workers int, count int) int {
	if workers <= 1 || count <= 1 {
		return 1
	}
	if workers > count {
		return count
	}
	return workers
}

// parallelForNodes runs fn over node ids [0,count) using deterministic
// contiguous partitions. Worker i receives the half-open interval
// [i*count/workerCount, (i+1)*count/workerCount). Integer division does not
// drop remainder nodes: adjacent intervals meet exactly, the first interval
// starts at 0, and the final interval ends at count. Each node id is therefore
// visited exactly once, and the loop condition nodeID < end prevents
// nodeID >= count.
//
// Errors are returned in node-id order after all workers finish. Callers must
// ensure fn writes only node-local output or worker-local scratch.
func parallelForNodes(count int, workers int, fn func(workerID int, nodeID int) error) error {
	workerCount := effectiveWorkerCount(workers, count)
	if workerCount == 1 {
		for nodeID := 0; nodeID < count; nodeID++ {
			if err := fn(0, nodeID); err != nil {
				return err
			}
		}
		return nil
	}

	errs := make([]error, count)
	var wg sync.WaitGroup
	for workerID := 0; workerID < workerCount; workerID++ {
		start := workerID * count / workerCount
		end := (workerID + 1) * count / workerCount
		wg.Add(1)
		go func(workerID int, start int, end int) {
			defer wg.Done()
			for nodeID := start; nodeID < end; nodeID++ {
				if err := fn(workerID, nodeID); err != nil {
					errs[nodeID] = err
				}
			}
		}(workerID, start, end)
	}
	wg.Wait()

	for nodeID := 0; nodeID < count; nodeID++ {
		if errs[nodeID] != nil {
			return errs[nodeID]
		}
	}
	return nil
}
