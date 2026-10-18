"""VLA training-data attribution PoC (docs/VLA_Data_Attribution_PoC.pdf).

Data layer (LIBERO -> LeRobot v3 dataset, episode catalog) and lineage layer (dataset versions,
splits, subsets). Grounding: PoC Key takeaways p.2, "Lineage (exact data versions, episode IDs,
configs and checkpoints tied to eval results) is the foundation"; §2.1 item 1 p.3; §2.2 p.4.

Modules, in the order data flows through them (docs/dataflow.md, "Full flow"):

- ``libero_convert``  raw LIBERO HDF5 -> one LeRobot v3 dataset (S1, the "LIBERO demos" box)
- ``catalog``         stable ``episode_id`` + per-episode ``content_hash`` + the §2.2 catalog (S2)
- ``lineage``         content-addressed dataset versions, reference/candidate splits, subsets and
                      the selection ops; also the ``python -m vlapoc.lineage`` CLI (S3)
- ``derive``          derived versions with ground-truth labels: the injection hook (S4, the
                      "corrupt 5-10% of a copy" box)
- ``runs``            every stage as an MLflow run, and the ``trace``/``consumers``/``diff`` queries (S5)
- ``lerobot_train_entry``  subprocess entry that calls LeRobot's own ``train()`` (S5)
- ``influence``       projected per-sample gradient features of a checkpoint (Step 3 input)
- ``attribution``     CUPID influence of every training episode on every family-W return: the
                      ``attribute`` stage that ``runs.select_run`` consumes for Step 4

``scripts/lineage_dry_run.py`` runs every stage type end to end (S6).
"""
