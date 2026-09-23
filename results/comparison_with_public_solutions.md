# Comparison with public solutions

After all of our outputs were final (and compared with the answer key), I looked at public GitHub repositories with solutions to the same assignments, to see how our methods and results compare and whether anything could be improved. The repositories have no license, so no code was copied; the comparison uses their committed output files and reads their methods. The scores are reproduced by

```
python scripts/compare_public_solutions.py
```

which clones each repository at the commit listed below and writes [`public_solutions_tables.md`](public_solutions_tables.md).

## Repositories

Five repositories use the same Fall 2025 data and include output files for the unknown sets:

| Repository (commit) | Authors as stated | Assignments, language | Closest-point search | ICP stopping rule | PA5 |
|---|---|---|---|---|---|
| [mayasharma604/CIS-PA3](https://github.com/mayasharma604/CIS-PA3) (93acba5), [CIS-PA4](https://github.com/mayasharma604/CIS-PA4) (c6c1537), [CIS-PA5](https://github.com/mayasharma604/CIS-PA5) (13531f6) | Maya Sharma, Anishka Bhartiya | PA3 to PA5, Python | brute force, region-based closest point on a triangle (Voronoi region tests) | mean distance changes by less than 1e-5, at most 50 iterations | code only, no outputs committed |
| [vibhakamath23/CIS-PA3](https://github.com/vibhakamath23/CIS-PA3) (f153df4) | | PA3, MATLAB | octree and bounding spheres, with unit tests for both | | |
| [pranhav16/CIS_PA4](https://github.com/pranhav16/CIS_PA4) (4ac772e) | Luiza Brunelli, Pranhav Sundararajan | PA4, MATLAB | brute force | mean distance changes by less than 1e-7, at most 25 iterations | |
| [SIDR73/cis2025](https://github.com/SIDR73/cis2025) (335eb89) | | PA2 to PA5, Python | KD-tree over the mesh vertices, testing the triangles around each visited vertex | the lecture's rule: match threshold 3 x mean error, stop when the ratio of successive mean errors stays in [0.95, 1] for 3 iterations, at most 250 | alternating weight least squares and rigid steps |

Older repositories (other years' data, so only the methods can be compared) include [SeanSDarcy2001/CISProgrammingAssignments](https://github.com/SeanSDarcy2001/CISProgrammingAssignments) (Fall 2021, d4f7f63), which has a covariance tree for the closest-point search, built on the course template [benjamindkilleen/ciscode](https://github.com/benjamindkilleen/ciscode), and [suryanshshukla10/CIS-PA4](https://github.com/suryanshshukla10/CIS-PA4) and [CIS-PA5](https://github.com/suryanshshukla10/CIS-PA5) (Fall 2021).

## Accuracy on the unknown sets

All solutions are scored like ours in [`answer_key_comparison.md`](answer_key_comparison.md): F_reg is recovered from each output file and compared with the actual F_reg in the instructor's log, and the sum of squared distances (SSE) from F_reg d_k to the surface measures how well each F_reg fits the data. Full tables: [`public_solutions_tables.md`](public_solutions_tables.md).

**PA3.** All four solutions give the same d_k and c_k to the 0.01 mm rounding of the files.

**PA4** (rotation error in degrees, translation error in mm, SSE in mm^2):

| Set | Ours | mayasharma604 | pranhav16 | SIDR73 |
|---|---|---|---|---|
| G (noise-free) | 0.0021, 0.0015; SSE 0.0038 | 0.0060, 0.0011; 0.0039 | 0.0525, 0.0027; 0.0147 | 0.0083, 0.0022; 0.0042 |
| H (noise-free) | 0.0052, 0.0017; 0.0034 | 0.0036, 0.0019; 0.0034 | 0.0194, 0.0107; 0.0101 | 0.0051, 0.0016; 0.0034 |
| J (0.1 mm noise) | 0.0575, 0.0073; 1.4909 | 0.0212, 0.0063; 1.4984 | 0.0410, 0.0072; 1.4920 | 0.0563, 0.0113; 1.4977 |
| K (0.1 mm noise) | 0.0742, 0.0153; 1.3115 | 0.0710, 0.0150; 1.3116 | 0.0688, 0.0151; 1.3118 | 0.0501, 0.0259; 1.3327 |

**PA5** (only SIDR73 published unknown-set outputs; max weight error in the last value):

| Set | Ours | SIDR73 |
|---|---|---|
| G (noise-free) | 0.0051 deg, 0.0018 mm; SSE 0.0030; weights 0.092 | 0.0700 deg, 0.1012 mm; 0.4011; 2.047 |
| H (noisy) | 0.0357, 0.0094; 2.6444; 0.599 | 0.0314, 0.0145; 2.6518; 0.459 |
| J (noisy) | 0.0359, 0.0171; 2.8058; 0.474 | 0.0459, 0.0170; 2.8407; 0.850 |
| K (noise-free) | 0.0032, 0.0022; 0.0024; 0.065 | 0.0471, 0.0114; 0.0713; 0.641 |

Observations:

- **Our F_reg has the lowest SSE on every PA4 and PA5 set**, so it is the best least-squares fit to the given data. On the noisy sets another solution is sometimes closer to the true F_reg (for example mayasharma604 on PA4-J), but with a higher SSE: with 0.1 mm noise, the distance to the truth varies by a few hundredths of a degree from one correct fit to another, and no solution is consistently closer.
- **pranhav16 stops after at most 25 iterations**, before convergence on the noise-free sets (rotation errors of 0.02 to 0.05 degrees, SSE three to four times ours).
- **SIDR73's vertex KD-tree does not always find the closest point.** The tree prunes with the coordinates of the vertices, but a triangle can extend beyond its vertices' splitting planes. Checking each written c_k against the true closest point to its s_k shows misses of up to 0.65 mm in PA4 and 1.1 mm in PA5 (2 to 6 samples per set), which explains its larger PA5 errors on the noise-free sets. For ours, mayasharma604 and pranhav16, every c_k is the true closest point to within the 0.01 mm rounding. This is why our tree is tested to give exactly the brute-force distances on every data set.
- **Speed.** mayasharma604 and pranhav16 use brute force with a Python or MATLAB loop over triangles; ours vectorizes brute force and adds the tree, which is 65 to 120 times faster than our own vectorized brute force on the pointer tips of the data sets. A full PA4 run of all 14 sets takes about 3 seconds.

## What was tried and changed

The comparison did not reveal an accuracy problem in our solution, but two methods from the other repositories were worth testing.

1. **Covariance (oriented-box) tree**, the idea of SeanSDarcy2001's `covtree.py`, from the lecture notes on finding point pairs. Their implementation was not reused (it builds the scatter matrix about the origin rather than the centroid, and does not lower the bound during the search). I added the option `BoundingBoxTree(mesh, oriented=True)`, in which each node's box is aligned with the principal axes of its triangle corners. The boxes enclose about a third of the volume, the results are identical to brute force, but on this mesh it is not faster (see [`search_timing.md`](search_timing.md)), so the axis-aligned tree stays the default.
2. **Tighter seed bound.** The covariance-tree experiment showed that the greedy seed of a search without a hint can be loose (4.1 mm on average against a true distance of 0.9 mm for PA4-B with the oriented tree). The search now first tests each query's nearest candidate leaf, tightens the bound and drops the leaves that are too far, before the remaining exact tests. For queries without a hint this cuts the triangle tests by about a third (for random points, 44 to 28 per query). With a hint from the previous ICP match the bound is already tight, so the step is skipped there; PA4 run times are unchanged. All output files are byte-for-byte unchanged, since the search was and is exact.

Region-based closest point on a triangle (mayasharma604, pranhav16) and octrees with bounding spheres (vibhakamath23) are correct alternatives to our methods but would not change our results; our tests already cover the obtuse-triangle case that the region tests are designed for.
