"""Method-side code: screening and attribution methods under test.

Grounding: PoC §2.1 item 4 (p. 3), "check whether your attribution method flags exactly those". Modules here
read data only through ``vlapoc.method_data.open_for_method`` and never import ``vlapoc.groundtruth``
(ground-truth feature S3, owner decision 3; enforced by ``scripts/verify_method_boundary_s3.py``).
"""
