# PA3 validation on the debug sets

Our output files in `output/` are compared with the instructor's `PA3-X-Debug-Output.txt` and `PA3-X-Debug-Answer.txt` files. The tables are printed by

```
python -m cisreg.pa3 --all
python -m cisreg.validate PA3
```

## Error measures

For sample k, with our pointer tip d_k and closest point c_k, and the reference values d'_k and c'_k from the file:

- e_d,k = |d_k - d'_k| (pointer tip, 3D distance in mm)
- e_c,k = |c_k - c'_k| (closest point on the mesh)
- e_dist,k = | |d_k - c_k| - |d'_k - c'_k| | (difference of the distance magnitudes)

Each table gives the maximum over the samples of a set and the root mean square, RMS = sqrt(mean_k e_k^2). All values are in mm.

## Results

Against the Output files:

| Set | N | max e_d | RMS e_d | max e_c | RMS e_c | max e_dist |
|---|---|---|---|---|---|---|
| A-Debug | 15 | 0.0141 | 0.0082 | 0.0141 | 0.0073 | 0.0070 |
| B-Debug | 15 | 0.0100 | 0.0077 | 0.0141 | 0.0068 | 0.0080 |
| C-Debug | 15 | 0.0141 | 0.0089 | 0.0141 | 0.0086 | 0.0070 |
| D-Debug | 15 | 0.0224 | 0.0110 | 0.0200 | 0.0089 | 0.0070 |
| E-Debug | 15 | 0.0200 | 0.0093 | 0.0141 | 0.0073 | 0.0100 |
| F-Debug | 15 | 0.0141 | 0.0089 | 0.0173 | 0.0077 | 0.0130 |

The PA3 Answer files contain exactly the same data lines as the Output files (only the file name in the header differs), so the table against the Answer files is identical and is not repeated.

For reference, our distances |d_k - c_k| have these means and maxima (mm): A 0.002 and 0.007, B 1.466 and 3.061, C 0.773 and 1.958, D 1.107 and 3.478, E 2.177 and 4.212, F 1.404 and 3.082. In set A the sample points are on the surface, as expected for a noise-free set with F_reg = I.

## Closest-point step on its own

To separate the matching from the pose computation, the closest point to each reference tip d'_k was found with the brute-force search and compared with the reference c'_k. The first column checks that the reference closest points lie on the mesh that was read.

| Set | max dist(c'_k, mesh) | max e_c | RMS e_c |
|---|---|---|---|
| A-Debug | 0.0048 | 0.0048 | 0.0024 |
| B-Debug | 0.0053 | 0.0093 | 0.0064 |
| C-Debug | 0.0060 | 0.0103 | 0.0064 |
| D-Debug | 0.0048 | 0.0101 | 0.0062 |
| E-Debug | 0.0051 | 0.0095 | 0.0068 |
| F-Debug | 0.0041 | 0.0108 | 0.0070 |

Every reference c'_k is within 0.006 mm of our mesh, and given the same d'_k our closest points agree with the reference to about 0.01 mm. Both numbers are at the level of the 0.01 mm rounding of the files (a point rounded to 0.01 mm per coordinate is up to 0.0087 mm from the true point).

## Interpretation

All differences are 0.022 mm or less, and the RMS differences are 0.007 to 0.011 mm. This is the size expected from rounding alone:

- Both our files and the reference files round coordinates to 0.01 mm. For two independent roundings of the same value, each coordinate difference has variance 2 (0.01^2 / 12), so the 3D RMS is sqrt(3 x 2 x 0.01^2 / 12) = 0.0071 mm.
- The input readings are also rounded to 0.01 mm. If the instructor computed d'_k from unrounded readings, this adds error to our d_k. A Monte Carlo test (adding uniform noise of plus or minus 0.005 mm to every reading of sets A, D and E, 200 trials) gives an RMS change of 0.0073 mm in d_k; the 100 mm lever arm of the pointer amplifies small rotation errors.
- Together these predict an RMS e_d of about sqrt(0.0071^2 + 0.0073^2) = 0.010 mm, which matches the observed 0.008 to 0.011 mm.

Sets E and F have 0.5 mm marker noise according to the instructor's notes, yet they agree as closely as the noise-free sets. That is expected: the noise is already in the readings, and both programs process the same readings. Noise changes where d_k is, not whether two correct programs agree on it.

There is no sign of a systematic error in the pose estimation, the tip computation or the closest-point search.
