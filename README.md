# Surface Registration: ICP and Deformable Registration to a Statistical Shape Model

My implementation of Programming Assignments 3, 4 and 5 from **Computer Integrated Surgery I** at Johns Hopkins University. It registers points touched on a bone with a tracked pointer to a CT-derived bone surface. The progression is: closest points on a triangle mesh, then rigid iterative closest point (ICP), then deformable registration to a statistical shape atlas. Everything is built from scratch and validated against the course's reference outputs.

This is the follow-up to [cis-surgical-navigation](https://github.com/mzlumi/cis-surgical-navigation), which covers PA1 and PA2 (frames, point-set registration, pivot calibration, EM distortion correction).

## The course

**EN.601.455/655 Computer Integrated Surgery I** is taught by Prof. Russell H. Taylor in the Department of Computer Science at Johns Hopkins, through the Laboratory for Computational Sensing and Robotics (LCSR). Taylor led early work on robot-assisted orthopaedic surgery at IBM Research and directed the NSF Engineering Research Center for Computer-Integrated Surgical Systems and Technology (CISST ERC) at Hopkins.

The course covers imaging, segmentation and modeling, frames and calibration, registration, tracking and navigation, and surgical robot systems. Five programming assignments run through one simulated scenario:

| Assignment | Topic | Repository |
|---|---|---|
| PA1 | Frame transformations, point-set registration, pivot calibration | cis-surgical-navigation |
| PA2 | EM distortion correction, EM-to-CT registration, navigation | cis-surgical-navigation |
| **PA3** | Closest point on a triangle mesh (the matching step of ICP) | this one |
| **PA4** | Full rigid ICP registration of probed points to a bone surface | this one |
| **PA5** | Deformable registration to a statistical shape model (optional in the course) | this one |

### Links

- Course page: <https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:455-655>
- Fall 2025 schedule, with handouts, data and lectures: <https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:2025:fall-2025-schedule>
- cisst libraries mentioned in the handouts: <https://github.com/jhu-cisst/cisst>

## The problem

A bone has been scanned with CT, and its surface is given as a triangle mesh. Two optically tracked rigid bodies are used:
- **body B** is screwed into the bone, so it moves with it;
- **body A** is a pointer whose tip touches points on the bone surface.

For each sample k, the poses F_A,k and F_B,k come from point-set registration of the LED readings. The pointer tip in bone-body coordinates is then d_k = F_B,k⁻¹ · F_A,k · A_tip.

- **PA3:** with F_reg = I, find the closest point c_k on the mesh to each s_k = F_reg · d_k. Start with a brute-force search, then add a faster spatial data structure (bounding spheres or a box tree) and check it against the brute force.
- **PA4:** iterate matching and rigid registration (ICP) until F_reg converges, with sensible stopping and outlier handling, and report s_k, c_k and the residuals.
- **PA5:** the bone is no longer fixed. Its shape is the mean shape plus a weighted sum of atlas modes. Estimate F_reg and the mode weights λ together, either by alternating rigid steps and mode steps, or with the linearized combined update described in the handout.

The full statements, file formats and rubrics are in [`docs/handout/`](docs/handout). Two course lectures used as references are in [`docs/reference/`](docs/reference).

## Data and validation

[`data/`](data/README.md) holds the official Fall 2025 data.
- **Debug sets** come with the instructor's outputs and the answers used to generate them.
- **Unknown sets** come without outputs.

The instructor's log files contain the true F_reg and mode weights for every set. They are kept as an answer key and used only after results are final.

Correctness is shown by matching the debug outputs (closest points, F_reg, mode weights), and by checking the fast search against brute force. The noisy sets are analysed against the stated marker noise.

## Status

- [ ] Data readers and Cartesian math (frames, rigid point-set registration) with tests
- [ ] Closest point on a triangle, brute-force closest point on a mesh
- [ ] Fast closest-point search, validated against brute force
- [ ] PA3 pipeline and outputs for all sets
- [ ] PA4 rigid ICP and outputs for all sets
- [ ] PA5 deformable registration and outputs for all sets
- [ ] Error analysis and report

## Credit

The problem, the handouts, the lecture notes and the data are by Russell H. Taylor and the CIS I teaching staff at Johns Hopkins University. They are included here for reference only. The code here was written for this project from the handouts alone; no code from past students' solutions was used.
