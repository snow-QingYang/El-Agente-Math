## LINE 249

Here, we focus on incremental data valuation, while the optimization for decremental data valuation follows a similar approach, which is detailed in the Appendix. Let the original dataset be $\mathcal { D }$ , containing $N$ samples, and the new data to be added be $\mathcal { D } ^ { \prime }$ , with $N ^ { \prime }$ samples. The augmented dataset is denoted as $\hat { \mathcal { D } } = \mathcal { D } \cup \mathcal { D } ^ { \prime }$ , and let $\beta ^ { c u r }$ represent the original data values in $\mathcal { D }$ .

In contrast to the only existing research on dynamic data valuation [46], which relies on recalculating Shapley values, this study investigates an alternative path that avoids the need to re-estimate Shapley values, thereby improving efficiency. Specifically, we aim to explore whether it is possible to infer the values of all data in $\hat { \mathcal { D } }$ based solely on the dataset $\hat { \mathcal { D } }$ and the original data values, $\beta ^ { c u r }$ .

As empirically analyzed in Section 3, the changes in value
