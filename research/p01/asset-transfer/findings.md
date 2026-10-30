# Findings

- Weight: /home/fengyi/sam3.pt, 3,450,062,241 bytes, frozen SHA256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e.
- LoveDA: /home/fengyi/E/data/SegEarth/LoveDA; validation images 1,669 / 2,411,772,953 bytes; validation GT 1,669 / 15,695,837 bytes.
- iSAID: /home/fengyi/E/data/SegEarth/iSAID; validation images 11,644 / 28,044,597,288 bytes; validation GT 11,644 / 37,047,487 bytes. GT filenames have _instance_color_RGB suffix, but existing audit says mode L; preserve bytes and names, do not transform labels.
- Existing inference and paired-asset manifests: /home/fengyi/CodeHome/ProC-SAM3-repro/manifests. Keep inference manifests separate from evaluation metadata.
- Required main payload ~31.63 GiB before tar overhead; R has ~145 GiB free. No environment archive needed under the user's prior revision.
- No existing P01 input split directory. Verified source-scene map still absent; final calibration/pilot/confirmation manifests cannot be honestly generated now.
- Old absolute paths and shipped /data paths can both point to one extracted asset copy. New machine may require sudo solely to create these path aliases.
- No graphify graph exists; this is an asset migration task, so no graph generation/API is needed.

Read-only reconnaissance encountered missing optional data/inputs paths, then located real roots from manifests; no modifications to external assets.
