# Course data

The official Fall 2025 data for Programming Assignments 3, 4 and 5 of JHU EN.601.455/655 Computer Integrated Surgery I (`2025_pa345_student_data.zip` from the course schedule page, <https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:2025:fall-2025-schedule>). The files are unchanged, with their original names. All coordinates are in millimetres.

## Shared inputs

| File | Contents |
|---|---|
| `Problem3Mesh.sur` (same file as `Problem3MeshFile.sur`, `Problem4MeshFile.sur`, `Problem5MeshFile.sur`) | Bone surface mesh in CT coordinates: 1568 vertices, then triangles as 3 vertex indices plus 3 neighbour triangle indices (-1 means no neighbour; not needed) |
| `ProblemN-BodyA.txt`, `ProblemN-BodyB.txt` (N = 3, 4, 5) | Rigid body definitions: LED marker positions in body coordinates, then the tip position. Body A is the pointer. Body B is screwed into the bone. The bodies differ between problems. |
| `Problem5Modes.txt` | Statistical shape atlas: mode 0 is the mean shape (same vertices as the mesh), then modes 1 to 6 are per-vertex displacements |

## Sample sets

Each set has `SampleReadingsTest.txt`, the LED readings for A markers, B markers and dummy markers per sample frame. Debug sets also have `Output.txt` (the instructor's program output) and `Answer.txt` (the values used to generate the data; they can differ slightly from the output because of simulated noise).

| Assignment | Debug sets | Unknown sets | Notes from the instructor log files |
|---|---|---|---|
| PA3 | A to F (15 samples in A) | G, H, J | E, F, H and J have 0.5 mm marker noise |
| PA4 | A to F, plus Demo-Fast and Demo-Slow variants of A and B | G, H, J, K | E, F, J and K have 0.1 mm marker noise; the instructor's ICP took about 2 to 100 iterations |
| PA5 | A to F | G, H, J, K | E, F, H and J have 0.1 mm noise; the sample header's last field gives the number of modes to use |

Output formats:
- **PA3 and PA4:** each line has d_k (or s_k for PA4), then the closest point c_k, then |d_k − c_k|.
- **PA5:** the first data line holds the solved mode weights, and the lines after it follow the PA4 format.

## Answer key: use only at the end

`PA3-Logfile.txt`, `PA4-Logfile.txt` and `PA5-Logfile.txt` are the instructor's run logs for **every** set, including the unknown ones. They list the noise level, iteration counts, RMS residual, the computed and actual F_reg, and (for PA5) the solved and actual mode weights. Treat them as an answer key: validate on the debug sets first, and compare unknown-set results against the logs only after the results are final.

## Quirks to handle when parsing

- Separators vary: the sample headers use commas (`16, 150, PA5-A-Debug-SampleReadingsTest.txt 6`), while the output and answer headers use spaces (`15 PA3-A-Debug-Answer.txt 0`). Split on commas and whitespace.
- The PA5 sample header has a fourth field, the number of modes. The PA3 and PA4 headers carry a trailing `0` there.
- The handout calls the files `paV-X-ddddd-SampleReadings.txt`. The actual names are `PAV-X-Ddddd-SampleReadingsTest.txt`.
