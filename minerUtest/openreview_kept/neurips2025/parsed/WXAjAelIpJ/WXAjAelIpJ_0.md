## LINE 64

two sets $A , B$ , we denote their symmetric difference by $A \triangle B = ( A \cup B ) \setminus ( A \cap B )$ .

# 2.1 Binary Vectors

7 We consider a space of binary vectors of arbitrary (but finite) length. Each such vector, $x \in \{ 0 , 1 \} ^ { n }$ ,   
can equivalently be viewed as a subset of coordinates on which there are ones in the vector, which   
we denote by $\dot { A } _ { x } = \{ i \in [ n ] : x _ { i } = 1 \}$ . For a permutation $\pi : [ n ] \to [ n ]$ , by $\pi ( x )$ we denote the   
60 vector $x$ with changed order of coordinates, i.e., $\pi ( x ) _ { i } = x _ { \pi ( i ) }$ for every $i \in [ n ]$ . Drawing from game   
1 theory notation, for vector $x \in \{ 0 , 1 \} ^ { n }$ , coordinate $i \in [ n ]$ , and $b \in \{ 0 , 1 \}$ we write $y = ( x _ { - i } , b )$ to   
denote a vector such that $y _ { i } = b$ and $y _ { j } = x _ { j }$ for every $j ^ { \bar { \mathbf { \alpha } } } \in [ n ] \setminus \{ i \}$ . For vector $x \in \{ 0 , 1 \} ^ { n }$ by $\bar { x }$   
we denote its bitwise negation, i.e., $\bar { x } = ( 1 - x _ { 1 } , \ldots , 1 - x _ { n } )$ . By $\circ$ we denote concatenation of   
vectors, i.e., for $x \in \{ 0 , \bar { 1 } \} ^ { n }$ and $y \in \{ 0 , 1 \} ^ { m }$ , if $z = x \circ y$ , then $z _ { i } = x _ { i }$ for $i \in [ n ]$ and $z _ { i } = y _ { i }$ for   
every $i \in [ m + n ] \setminus [ n ]$ . Finally, by $x ^ { k }$ we denote the result of concatenating $k$ copies of $x$ one after   
another. For convenience, we will allow for $x ^ { 0 }$ to be the empty vector, which concatenated with any   
other vector $y$ , results in $y$ .

# 2.2 Dissimilarity Measures

A dissimilarity measure is a function that takes a pair
