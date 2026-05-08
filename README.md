Segment named objects (pillows, lamps, blanket etc.) from a 3D obj file using Florence-2 grounding, depth lifting, and multi-view voting.

The 3D mesh is loaded and 200,000 points are sampled evenly across its surface.

Florence-2 is run on 3 quick views to measure how big the target object appears. This auto-profiling step sets the detection thresholds automatically. 
    - min_votes: How many views a point must appear in to survive.
    - min_area, max_area: how small or large a detected box can be.
    - depth_zscore: explained later. 

The point cloud is projected flat onto a 2D image from one of 8 angles, with each point colored by its depth. (Depth based veridis colouring was used to assists florence with the grounding 
work)

That image is passed to Florence-2, which draws bounding boxes around anything matching the target label.

Boxes that are too small (noise) or too large (whole scene) are discarded.

DepthAnything V2 estimates depth for every pixel in the image. Pixels inside the box that are at a very different depth than the target surface are cut out to get cleaner segments. How 
strictly the depth mask cuts out background pixels inside a box is controlled by depth_zscore. 

The surviving pixels are looked up in an index map to find which 3D points they came from and those points receive a vote.

After having this done for all 8 views, any 3D point that collected enough votes across views is kept as part of the segmented object.

The result is visualized with segmented points in red and everything else in grey.

<img width="1048" height="825" alt="image" src="https://github.com/user-attachments/assets/4c9c0057-d42f-4868-8ece-4a327b7691d9" />
for prompt "pillow"

<img width="869" height="467" alt="image" src="https://github.com/user-attachments/assets/b28beadc-1e2f-4620-89c8-42d3c24b3d47" />
for prompt "lamp"

