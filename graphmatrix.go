package graphmatrix

// graphmatrix provides csr sparse matrices up to 2^32-1 rows/columns.

import (
	"errors"
	"fmt"
	"sort"
)

//Define en empty sentinel in the indexes into indices slice
var EmptySentinel = ^uint64(0)

// GraphMatrix holds a row index and vector of column pointers.
// If a point is defined at a particular row i and column j, an
// edge exists between vertex i and vertex j.
// GraphMatrices thus represent directed graphs; undirected graphs
// must explicitly set the reverse edge from j to i.
type GraphMatrix struct {
	IndPtr  []uint64 // indexes into Indices - must be twice the width of Indices.
	Indices []uint32 // contains the row values for each column. A stride represents the outneighbors of a vertex at col j.
}

// NZIter is an iterator over the defined points in the graphmatrix.
// If NZIter.Done is true, there are no more points defined.
// Changing the graphmatrix in the middle of an iteration will lead
// to undefined (and almost certainly unwanted) behavior.
type NZIter struct {
	g           GraphMatrix
	Done        bool
	indPtrIndex uint32 // index into g.IndPtr
	indIndex    uint64 // index into g.Indices
}

func (it *NZIter) Next() (r, c uint32, done bool) {
	// if we're already done
	if it.Done {
		return 0, 0, true
	}
	// get the row and column
	r = it.indPtrIndex
	c = it.g.Indices[it.indIndex]

	// increment to the next set index
	it.indIndex++
	// did we move to a new row?
	if it.indIndex >= it.g.IndPtr[r+1] && it.g.IndPtr[r+1] != EmptySentinel {
		it.indPtrIndex++
	}
	done = it.indPtrIndex >= uint32(len(it.g.IndPtr)-1)
	it.Done = done
	return r, c, done
}

func (g GraphMatrix) NewNZIter() *NZIter {
	firstRow := uint32(0)
	for g.IndPtr[firstRow] == EmptySentinel {
		firstRow++
	}
	if firstRow != 0 {
		firstRow--
	}
	return &NZIter{g, false, firstRow, 0}
}

// NewGraphMatrix creates an m x m sparse matrix.
func New(m int) (GraphMatrix, error) {
	if m < 0 {
		return GraphMatrix{}, errors.New("dimensions must be non-negative")
	}
	i := make([]uint32, 0)
	ip := make([]uint64, m+1)

	for i := 0; i < m+1; i++ {
		ip[i] = EmptySentinel
	}
	return GraphMatrix{IndPtr: ip, Indices: i}, nil
}

func NewFromRC(Indices []uint32, IndPtr []uint64) (GraphMatrix, error) {
	return GraphMatrix{IndPtr: IndPtr, Indices: Indices}, nil
}

func (g GraphMatrix) String() string {
	return fmt.Sprintf("GraphMatrix %v, %v, size %d", g.IndPtr, g.Indices, g.Dim())
}

// returns the maximum uint and its position in the vector.
// returns -1 as position if the vector is empty.
func maxUint32(v []uint32) (max uint32, maxPos int) {
	if len(v) == 0 {
		return 0, -1
	}
	max = 0
	maxPos = 0
	for i, n := range v {
		if n > max {
			max = n
			maxPos = i
		}
	}
	return max, maxPos
}

// NewFromSortedIJ creates a graph matrix using i,j as src, dst
// of edges. Assumes i and j are already sorted, j first.
func NewFromSortedIJ(s, d []uint32) (GraphMatrix, error) {
	if len(s) != len(d) {
		return GraphMatrix{}, fmt.Errorf("graph inputs must be of the same length (got %d, %d)", len(s), len(d))
	}
	m1 := s[len(s)-1]     // max s - this is O(1)
	m2, _ := maxUint32(d) // max d - this is O(n)
	m := m1
	if m2 > m1 {
		m = m2
	}
	m++

	//Allocate the slice with a capacity and a length of m elements
	IndPtr := make([]uint64, m)
	currval := uint32(0)
	//Init all the values with a default value
	for i := 0; uint32(i) < m; i++ {
		IndPtr[i] = EmptySentinel
	}
	first := true
	for i, n := range s {
		if first || n > currval { // the row has changed
			IndPtr[n] = uint64(i)
			currval = n
			first = false
		}
	}
	IndPtr = append(IndPtr, uint64(len(d)))
	return GraphMatrix{IndPtr: IndPtr, Indices: d}, nil
}

// SortIJ sorts two vectors s and d by s, then by d, and eliminates any duplicate pairs.
// Modifies s and d.
func SortIJ(s, d *[]uint32) error {
	if len(*s) != len(*d) {
		return errors.New("inputs must be of the same length")
	}
	sd := make([]uint64, len(*s))
	for i := 0; i < len(*s); i++ {
		sd[i] = uint64((*s)[i])<<32 + uint64((*d)[i])
	}
	sort.Slice(sd, func(i, j int) bool { return sd[i] < sd[j] })
	UniqSorted(&sd)

	for i := 0; i < len(sd); i++ {
		(*s)[i] = uint32(sd[i] >> 32)
		(*d)[i] = uint32(sd[i] & 0x00000000ffffffff)
	}

	(*s) = (*s)[:len(sd)]
	(*d) = (*d)[:len(sd)]
	return nil
}

func UniqSorted(a *[]uint64) {
	j := 0
	for i := 1; i < len(*a); i++ {
		if (*a)[j] == (*a)[i] {
			continue
		}
		j++
		// preserve the original data
		// in[i], in[j] = in[j], in[i]
		// only set what is required
		(*a)[j] = (*a)[i]
	}
	(*a) = (*a)[:j+1]

}

// inRange returns true if (r, c) is a valid index into v.
func (g *GraphMatrix) inRange(r, c uint32) bool {
	n := g.Dim()
	return (c < n) && (r < n)
}

// Dim returns the (single-axis) dimension of the GraphMatrix
func (g *GraphMatrix) Dim() uint32 {
	return uint32(len(g.IndPtr) - 1)
}

// N returns the number of defined values in the GraphMatrix
func (g *GraphMatrix) N() uint64 {
	return uint64(len(g.Indices))
}

// SearchSorted32 finds a value x in sorted vector v.
// Returns index and true/false indicating found.
// lo and hi constrains search to these Indices.
// If lo/hi are out of bounds, return -1 and false unless
// the vector is empty, in which case return 0 and false.
func SearchSorted32(v []uint32, x uint32, lo, hi uint64) (int, bool) {
	ulen := uint64(len(v))
	if ulen == 0 {
		return 0, false
	}
	if lo == hi {
		return int(lo), false
	}
	if ulen < lo || ulen < hi || lo > hi {
		return -1, false
	}
	s := sort.Search(int(hi-lo), func(i int) bool { return v[int(lo)+i] >= x }) + int(lo)
	found := (s < len(v)) && (v[s] == x)

	return s, found
}

// GetIndex returns true if the value at (r, c) is defined.
func (g GraphMatrix) GetIndex(r, c uint32) bool {
	if uint32(len(g.IndPtr)) <= c+1 {
		return false
	}
	next := r + 1
	r1 := g.IndPtr[r]
	r2 := g.IndPtr[next]
	for {
		if r2 != EmptySentinel {
			break
		}
		next++
		r2 = g.IndPtr[next]
	}
	if r1 >= r2 {
		return false
	}
	_, found := SearchSorted32(g.Indices, c, r1, r2)
	return found
}

// GetRow returns the 'n'th row slice, or an empty slice if empty.
func (g GraphMatrix) GetRow(n uint32) []uint32 {
	p1 := g.IndPtr[n]
	//if we can't find a index for a given index,
	//it means the node does not exist or does not have neighbors
	if p1 == EmptySentinel {
		return nil
	}
	next := n + 1
	//p2 represents the upper bound in the indice slice
	p2 := g.IndPtr[next]
	//p2 can't be linked to an EmptySentinel.
	//it must me linked to an existing vertex in the graph
	for {
		if p2 != EmptySentinel {
			break
		}
		next++
		p2 = g.IndPtr[next]
	}
	leng := uint64(len(g.Indices))
	if p1 > leng || p2 > leng {
		return []uint32{}
	}
	return g.Indices[p1:p2]
}

// SetIndex sets the value at (r, c) to true.
// This can be a relatively expensive operation as it can force
// reallocation as the vectors increase in size.
func (g *GraphMatrix) SetIndex(r, c uint32) error {
	if !g.inRange(r, c) {
		return errors.New("index out of range")
	}
	prev := r
	rowStartIdx := g.IndPtr[prev] // this is the pointer into the Indices for column c

	for {
		if rowStartIdx != EmptySentinel {
			break
		}
		if prev > 0 {
			prev--
			rowStartIdx = g.IndPtr[prev]
		} else {
			rowStartIdx = 0
			break
		}
	}

	next := r + 1
	rowEndIdx := g.IndPtr[next]
	for {
		if rowEndIdx != EmptySentinel {
			break
		}
		next++
		if next < g.Dim() {
			rowEndIdx = g.IndPtr[next]
		} else {
			rowEndIdx = uint64(g.Dim() - 1)
			break
		}
	}
	fmt.Println("SetIndex", rowStartIdx, rowEndIdx)
	j, found := SearchSorted32(g.Indices, c, rowStartIdx, rowEndIdx)

	fmt.Println("SearchSorted32", c, j, found)

	if found { // already set
		return nil
	}
	g.Indices = append(g.Indices, 0)
	copy(g.Indices[j+1:], g.Indices[j:])
	g.Indices[j] = c
	g.IndPtr[c] = uint64(j)
	for i := int(next); i < len(g.IndPtr); i++ {
		if g.IndPtr[i] != EmptySentinel {
			g.IndPtr[i]++
		}
	}

	return nil
}
