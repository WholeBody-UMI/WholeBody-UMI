# Source, example and visualization asset provenance

Project inference source was extracted from the user's
`Motion_Prior_Manipualtion` workspace, whose root source license is Apache-2.0.
It implements the WholeBody-UMI Motion Prior inference path.

`examples/walk_clean_table.npz` is the original tracked
`pipeline/Prior_Recon/Masked_Flow/data_test/walk-clean-table/ep_00000.npz`.
Confirm its capture/dataset provenance and redistribution terms before public
publication. No controller source or SDK is included.

`wholebody_umi/assets/g1/` contains the robot XML and its referenced meshes
from the tracked `pipeline/Prior_Recon/Masked_Flow/assets/` directory.
The XML retains the G1 joint order and geometry, removes unused actuators and
sensors, and uses a white scene with a floor grid for visualization.
The available upstream README attributes the robot description to Unitree
Robotics and is preserved in `licenses/UNITREE-G1-UPSTREAM-README.md`.
The inspected source asset directory has no dedicated asset LICENSE; the
project source license does not establish mesh redistribution terms.
Confirm the applicable asset license before public redistribution.

The exported inference checkpoint comes from the user's training workspace;
it is excluded from Git and must be provided separately with the intended model
redistribution terms.
