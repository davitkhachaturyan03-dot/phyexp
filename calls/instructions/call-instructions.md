# Call instructions

Standing instructions for calls with people related to the startup. Each topic gets its own `##` section,
so new instructions can be added without touching the others. The `call-notes` skill reads this file when it
stores a call.

## Questions to ask to potential pilot cooperators

The method attributes **closed-loop rollout outcomes** back to training episodes: train a policy on their
data, run it, time its successes, trace them to episodes. So the file format matters less than whether we
can train on their data and then measure the trained policy. Ask in this order.

### 1. Embodiment and actions (can a policy be trained on it at all?)
- How was it collected: robot teleoperation, a handheld gripper, or a head-mounted camera on a human?
- Are actions recorded? In what space (joint or end-effector, absolute or delta), frame, units and rate?
- Is there proprioception and gripper state?
- Are camera intrinsics and extrinsics provided?

### 2. Evaluation access (the deal-breaker)
- Is there a robot or simulator matching the data, where a policy trained on it can be run?
- Can we run about 50 rollouts per task from fixed initial states? (PoC §2.1: "roughly 50 or more, fixed
  seeds and initial states".)
- Can success, and *when* it happened, be detected automatically? Can failures be told apart: dropped,
  stuck, timed out?
- Do same-fixture human reference runs exist? They anchor human-relative throughput (PoC §3.2).
- With no robot of their own, real-robot evaluation needs the PhAIL station or a replica built with
  positronic (PoC p. 8).

### 3. Episode integrity
- Was anything filtered, trimmed or dropped, and by what rule? (The public LIBERO set silently drops 189 of
  1000 demos and shortens 68 others.)
- How were success labels produced: a person, an automatic goal check, or assumed? (LIBERO's stored reward
  is hard-coded to 1 at the last step, so it proves nothing.)
- How were language annotations made, by people or by a model?
- Do episodes have stable IDs that survive re-exports? Are there duplicates?

### 4. Metadata (it powers per-episode breakdowns, PoC §2.2)
- Per episode: source, task, instruction, scene, objects, operator, collection date, device.

### 5. Scale and structure
- How many episodes, hours and tasks? How many demos per task? Per-task attribution and a held-out clean
  reference set (PoC §8.3) need several demos per task.
- What is the total size? Will more collection campaigns follow?

### 6. Rights and privacy (ask early)
- Can we **modify** the data? The ground-truth test corrupts a copy (PoC §2.5).
- Can we train on it and keep the trained models? May we use it commercially?
- Do faces or bystanders appear in egocentric video? Is there consent, and does it need anonymising?
- Check any licence terms carefully. For example, EgoDex is CC-BY-NC-ND; "no derivatives" and
  "non-commercial" may rule out a corrupted copy and a commercial pilot (my reading, not legal advice).

### 7. Compute and delivery
- The method retrains policies for its counterfactual checks. Who pays for training on their data, and where
  does it run?
- How is the data delivered: format, versioning, checksums?

### Data formats to ask for, in order of preference
1. **LeRobot v3** (Parquet states and actions, MP4 per camera, `meta/` with schema, fps, stats, tasks and
   episodes).
2. **RLDS / TFDS** (episodes of steps in TFRecord files, as in Open X-Embodiment).
3. **robomimic-style HDF5**, like LIBERO.
4. **Raw ROS 2 bags (MCAP).** Most work for us: episode segmentation, sync and choosing an action topic.

Whatever the container: per episode, unfiltered, with timestamps, actions in a documented frame and unit,
and calibration.

### How to read their answers
- **Green:** robot teleoperation data, plus a robot or simulator matching it where we can run rollouts.
- **Yellow:** handheld-gripper data plus a robot with that gripper, or egocentric human data *alongside* some
  robot data on a robot we can evaluate. That is the co-training case; the question becomes which human
  demos helped the robot policy.
- **Red, for outcome-based attribution:** egocentric human video only, with no robot to deploy on. Offline
  quality scoring (the DemInf baseline) is still possible, but it is the baseline to beat, not our method.

## Note for people collecting egocentric data

*Send or read out as is. Statements marked (inferred) are our reasoning, not measurements.*

**Egocentric recordings are not yet VLA training data, and that gap is a business bottleneck for you, not only
a technical one.** Since most egocentric data is collected to train VLAs (vision-language-action models), what
decides its value is how much of it a VLA team can actually use as a training target.

An egocentric recording, EgoDex for example, stores a head-camera video plus estimated human hand poses:
transforms for the camera and 25 joints per hand, each with an ARKit confidence between 0 and 1, in a frame
anchored to the room (quoted, EgoDex README). That's what a human did, not a command a robot can execute. To
use it as a training target, the buyer has to bridge four gaps. All four are inferred, and they're the
problems EgoMimic-style work addresses:

1. **Embodiment.** A human hand has about 20+ degrees of freedom; a parallel gripper has 1. Arm reach and
   kinematics differ too, so "the hand moved here" doesn't map uniquely onto a robot command.
2. **Retargeting.** You choose a mapping, such as wrist pose to end-effector pose and finger closure to
   gripper open/closed. That's a modelling choice, it loses information, and it bakes in your assumptions.
3. **No robot state.** There are no joint positions or gripper readings, so the policy's state input doesn't
   exist. You'd substitute estimated hand pose, which is noisy where confidence is low.
4. **Visual and viewpoint mismatch.** The camera moves with the head and the images show a human hand, while
   a robot sees from fixed or wrist cameras and sees its own arm. A policy trained only on human video sees
   images it will never see when deployed.

**Why this is your problem too (inferred).** Every gap is a cost the data's buyer pays before the first
training run, or a reason the trained policy fails on a real robot. Data that needs all four bridged is worth
less per hour than data that closes some of them at collection time. The more of your data a VLA team can use
directly, the more it's worth to them, and the more likely they are to come back for the next campaign.

**What would help (inferred):** ask us which of these gaps matter for the model you're collecting for, and
whether some robot-side data (state, actions, matching camera views) can be collected alongside the
egocentric recordings, since co-training human and robot data is how the published approaches get around the
gaps.
