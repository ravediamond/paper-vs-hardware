# The rig

| part | detail |
|---|---|
| Arm | SO-101 follower (STS3215 servos, position-controlled, no force sensing) + SO-101 leader for teleop |
| Cameras | Logitech C920 as `front_top_camera` (top view), icSpring USB camera as `wrist_camera`; both 640×480 @ 30 fps, OpenCV backend |
| Compute on the robot | NVIDIA Jetson Orin NX 16 GB (Super dev kit), JetPack 6.2 / L4T R36.4.3, CUDA 12.6 |
| Power modes | `MAXN_SUPER` + `jetson_clocks` for the 2026-09/10 gate runs and benchmarks stated as such; 25W (jetson_clocks off) for offline runs labelled 25W. Each post's README says which |
| Control | 30 Hz, LeRobot `lerobot-rollout`-style loop (episodic, sync unless stated), 25 s trials |
| Software | the `lerobot-jetson` container (torch 2.11.0 built for JetPack 6, torchcodec source build); exact LeRobot commits per post in `posts/*/env/` |
| Task | "Put the blue bowl in the pink bowl" / "Put the pink bowl in the blue bowl" on a green cutting mat |

Camera placement and calibration are not published as files (the calibration JSON is specific to this arm). The
profile shape, with device paths removed:

```json
{
  "cameras": [
    {"name": "front_top_camera", "type": "opencv", "width": 640, "height": 480, "fps": 30},
    {"name": "wrist_camera", "type": "opencv", "width": 640, "height": 480, "fps": 30}
  ]
}
```
