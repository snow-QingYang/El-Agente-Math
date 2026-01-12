## LINE 145

Layer-wise Cosine Similarity Score

Inspired by the Layer-Wise MultiKrum defense method (details in Appendix A), and employing   
cosine similarity as the core measurement(Section 3.1), we introduce a novel metric termed the   
Layer-wise Cosine Similarity Score (LCSS). Diverging from traditional approaches that analyze   
client updates at the network level, LCSS enables independent analysis of updates at each individual   
layer.   
Formally, let $\Delta w _ { i } ^ { l }$ denote the update of client $i$ at the $l$ -th layer, where $l \in [ L ]$ indexes the $L$ layers of   
the global model and $i , j \in { 1 , \dots , N }$ represent client indices among the total of $N$ participants. The   
cosine similarity matrix $S ^ { l }$ at layer $l$ is defined as:

$$
S _ { i , j } ^ { l } = \frac { \langle \Delta w _ { i } ^ { l } , \Delta w _ { j } ^ { l } \rangle } { \| \Delta w _ { i } ^ { l } \| _ { 2 } \cdot \| \Delta w _ { j } ^ { l } \| _ { 2 } }
$$

where $\langle \cdot , \cdot \rangle$ denotes the inner product and $\| \cdot \| _ { 2 }$ is the $\ell _ { 2 }$ -norm. This matrix
