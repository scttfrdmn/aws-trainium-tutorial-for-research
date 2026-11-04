<!-- Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0 -->

# RUNBOOK-A: the full Phase A sequence

Run only after RUNBOOK-A0 has reproduced the port's number within 10 % (#70). Every command is
run from `benchmarks/structure-prediction/` in a checkout of this repository unless it says
otherwise. Settings for every comparison run: Boltz-2 defaults (3 recycles, 200 sampling steps),
ds=1 and ds=5, fk_steering off, seeds 0 and 1, the same `.a3m` files on every platform.

Placeholders: `<BUCKET>` an S3 bucket you control for this project, `<KEY>` an EC2 key pair,
`<PROFILE>` an instance profile that can read and write `s3://<BUCKET>/sp/`, `<AZ>` an AZ that
offers the instance type. Do not write account IDs into issues or files.

Order matters. The MSAs come first, because every Boltz run (the oracle included) reads them.
The CPU oracle then starts as soon as they exist, at best day 2–3 after the database download,
and it is the long pole: every trn2 gate result waits on it. It runs as parallel pinned workers
on the c7a. Its layout and budget are calibrated on one ~300-residue target first, never
extrapolated from the toy run. Post the calibrated budget on #55, then restate the Phase A
duration there.

Primary rows use Boltz-2's default `--max_msa_seqs` everywhere. T1061 also runs at
`--max_msa_seqs 2048` on every platform as one labelled secondary row (`--variant msa2048`).
T1047s1 has no ground truth: it is run and timed like every other target, but not gated.

**Stage timing (end-to-end batch column).**
- The first command on every host after login is
  `PYTHONPATH=. python -m harness.stages boot --out ~/stages.json --role <role>`. It records the
  seconds since boot and the instance type.
- Wrap setup, database, cache and MSA commands as
  `python -m harness.stages time <name> --out ~/stages.json -- <command>`, using one of these
  names: `setup`, `db_build`, `db_preload`, `msa`, `cache_build`, `cache_restore`.
- Before teardown, copy `~/stages.json` to `s3://<BUCKET>/sp/stages/<role>.json`.

## Day 1 — databases, MSAs, oracle

1. Rebuild the ground-truth files locally (the committed `targets/targets.json` must not change
   except for `generated_at`; if it does, stop and comment on #60).

   ```bash
   python -m targets.build_targets --out targets --truth-dir ~/sp-data/truth
   git diff --stat targets/
   aws s3 sync ~/sp-data/truth s3://<BUCKET>/sp/truth/
   ```

2. Launch the g7e.2xlarge (2 TB gp3 data volume mounted at `/data`).

   ```bash
   python infra/launch.py plan   g7e --az <AZ> --key-name <KEY> --iam-instance-profile <PROFILE>
   python infra/launch.py launch g7e --az <AZ> --key-name <KEY> --iam-instance-profile <PROFILE> --yes
   ```

3. On the g7e: install ColabFold v1.6.3 and MMseqs2 18-8cc5c (GPU build), then download the
   databases once into S3 and build the GPU index (see `msa/RUNBOOK.md` for sizes and options).

   ```bash
   git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research.git ~/repo
   cd ~/repo/benchmarks/structure-prediction
   python3 -m venv ~/cf && source ~/cf/bin/activate && pip install "colabfold==1.6.3" boto3 pyyaml numpy
   export PYTHONPATH=$PWD; S="python -m harness.stages"
   $S boot --out ~/stages.json --role g7e
   $S time setup --out ~/stages.json -- bash -c 'wget -qO- https://mmseqs.com/archive/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/mmseqs-linux-gpu.tar.gz | tar xz -C ~'
   export PATH=~/mmseqs/bin:$PATH
   bash msa/stage_databases.sh upload  /data/colabfold s3://<BUCKET>/sp/colabfold-dbs
   $S time db_build --out ~/stages.json -- bash msa/stage_databases.sh restore /data/colabfold s3://<BUCKET>/sp/colabfold-dbs gpu
   ```

4. On the g7e: generate the MSAs (MMseqs2-GPU, one query at a time) and publish them. These
   `.a3m` files are the inputs for every platform; they are never regenerated per platform.

   ```bash
   $S time msa --out ~/stages.json -- bash msa/msa_gpu.sh /data/colabfold targets/targets.json /data/msas /data/msa_gpu_timing.json
   aws s3 sync /data/msas s3://<BUCKET>/sp/msas/
   aws s3 cp /data/msa_gpu_timing.json s3://<BUCKET>/sp/timing/
   aws s3 cp /data/msa_gpu_timing.manifest.json s3://<BUCKET>/sp/timing/
   ```

5. Launch the c7a.48xlarge for the CPU fp32 oracle (spawn; TTL 216 h).

   ```bash
   python infra/launch.py plan   c7a --az <AZ> --key-name <KEY> --iam-instance-profile <PROFILE>
   python infra/launch.py launch c7a --az <AZ> --key-name <KEY> --iam-instance-profile <PROFILE> --yes
   ```

6. On the c7a: install stock Boltz-2 2.2.1, pull the MSAs and write the Boltz inputs.

   ```bash
   git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research.git ~/repo
   cd ~/repo/benchmarks/structure-prediction && export PYTHONPATH=$PWD
   python3 -m venv ~/bz && source ~/bz/bin/activate && pip install "boltz==2.2.1" boto3
   aws s3 sync s3://<BUCKET>/sp/msas/ ~/msas/
   python -m harness.run inputs --targets targets/targets.json --msa-dir ~/msas --out ~/inputs
   ```

7. Calibrate the oracle as the first job on the c7a. Use T1042 (289 residues): run it solo on 24
   pinned threads, then as 8 concurrent pinned workers × 24 threads (192 vCPUs). This measures
   the per-worker slowdown before the layout is committed. The run also checks the dtype audit
   on a real target: every submodule float32 and no autocast, including in `structure_module`
   (the diffusion decoder).

   ```bash
   python -m harness.run oracle-calibrate --target-id T1042 --input ~/inputs/T1042.yaml --tokens 289 \
     --out-root results/oracle-calibration --workers 8 --threads 24 --repo this-repo=$HOME/repo
   python -m harness.oracle budget --calibration results/oracle-calibration/calibration.json \
     --targets targets/targets.json --workers 8 --extra T1061:msa2048 --host-ram-gib 384
   ```

   Post `calibration.json` and the budget on #55 (k=2 and k=3 bracket the scaling) and restate
   the Phase A duration there. If the slowdown is large or the memory projection does not fit,
   rerun the calibration at 4 × 48 or 6 × 32 and pick the layout with the shortest makespan.
   Comment the choice on #70.

8. Start the oracle with the chosen layout: the 19 gated CASP14 targets × seeds 0 and 1, largest
   first, then T1061 at `--max_msa_seqs 2048`. The ceiling complexes and T1047s1 get no oracle
   run. Results are cached under `results/oracle/` and never rerun unless the target set or the
   Boltz version changes; a run that fails the fp32 checks is marked `failed`, not cached.

   ```bash
   cat > ~/oracle.sh <<'EOF'
   cd ~/repo/benchmarks/structure-prediction && export PYTHONPATH=$PWD
   python -m harness.run oracle --targets-json targets/targets.json --inputs ~/inputs \
     --out-root results/oracle --workers 8 --threads 24 --repo this-repo=$HOME/repo
   python -m harness.run oracle --targets-json targets/targets.json --inputs ~/inputs --only T1061 \
     --variant msa2048 --max-msa-seqs 2048 --out-root results/oracle --workers 2 --threads 96
   aws s3 sync results/oracle s3://<BUCKET>/sp/results/oracle/
   EOF
   nohup bash ~/oracle.sh > ~/oracle.log 2>&1 &
   (while sleep 3600; do aws s3 sync ~/repo/benchmarks/structure-prediction/results/oracle s3://<BUCKET>/sp/results/oracle/; done) &
   ```

   Comment on #70 with the start time and layout.

## Day 1–2 — MSA timing for the trn2 pipeline (r7a)

9. Launch the r7a.48xlarge (1,536 GiB RAM, 2 TB gp3), build the CPU index from the same staged
   databases, time MMseqs2-CPU per target (output discarded), publish, tear down.

   ```bash
   python infra/launch.py launch r7a --az <AZ> --key-name <KEY> --iam-instance-profile <PROFILE> --yes
   # on the r7a:
   git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research.git ~/repo && cd ~/repo/benchmarks/structure-prediction
   python3 -m venv ~/cf && source ~/cf/bin/activate && pip install "colabfold==1.6.3" boto3 pyyaml numpy
   export PYTHONPATH=$PWD; S="python -m harness.stages"
   $S boot --out ~/stages.json --role r7a
   $S time setup --out ~/stages.json -- bash -c 'wget -qO- https://mmseqs.com/archive/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/mmseqs-linux-avx2.tar.gz | tar xz -C ~'
   export PATH=~/mmseqs/bin:$PATH
   $S time db_build --out ~/stages.json -- bash msa/stage_databases.sh restore /data/colabfold s3://<BUCKET>/sp/colabfold-dbs cpu
   $S time msa --out ~/stages.json -- bash msa/msa_cpu_timing.sh /data/colabfold targets/targets.json /data/msa_cpu_timing.json
   aws s3 cp /data/msa_cpu_timing.json s3://<BUCKET>/sp/timing/
   aws s3 cp /data/msa_cpu_timing.manifest.json s3://<BUCKET>/sp/timing/
   aws s3 cp ~/stages.json s3://<BUCKET>/sp/stages/r7a.json
   # back on the workstation:
   python infra/launch.py teardown r7a --yes
   ```

## Days 2–4 — g7e runs (Phase A task 2, #72)

10. On the g7e: install stock Boltz-2 2.2.1 with the cuEquivariance kernels and write the inputs.
   Run every CASP14 target and both ceiling complexes at ds=1 and ds=5, seeds 0 and 1, with a
   profiled extra run per configuration (its timings are discarded). Then run the T1061
   secondary row and the end-to-end batch directory runs.

   ```bash
   python3 -m venv ~/bz && source ~/bz/bin/activate && pip install "boltz[cuda]==2.2.1" boto3
   cd ~/repo/benchmarks/structure-prediction
   PYTHONPATH=. python -m harness.run inputs --targets targets/targets.json --msa-dir /data/msas --out ~/inputs
   for f in ~/inputs/*.yaml; do t=$(basename $f .yaml); for ds in 1 5; do for s in 0 1; do
     PYTHONPATH=. python -m harness.run predict --backend gpu --target-id $t --input $f \
       --out-dir results/raw/g7e/ds$ds/$t/seed$s --seed $s --ds $ds --profile \
       --targets-json targets/targets.json --repo this-repo=$HOME/repo
   done; done; done
   for ds in 1 5; do for s in 0 1; do   # secondary row
     PYTHONPATH=. python -m harness.run predict --backend gpu --target-id T1061 --input ~/inputs/T1061.yaml \
       --out-dir results/raw/g7e-msa2048/ds$ds/T1061/seed$s --seed $s --ds $ds \
       --variant msa2048 --max-msa-seqs 2048 --targets-json targets/targets.json
   done; done
   mkdir -p ~/inputs-casp && for t in $(python -c "import json;print(' '.join(t['id'] for t in json.load(open('targets/targets.json'))['targets'] if t['set']=='casp14'))"); do cp ~/inputs/$t.yaml ~/inputs-casp/; done
   for ds in 1 5; do for s in 0 1; do   # end-to-end batch: the 20-target directory, one process
     PYTHONPATH=. python -m harness.run directory --backend gpu --inputs ~/inputs-casp \
       --out-dir results/raw/g7e-dir/ds$ds-s$s --seed $s --ds $ds --repo this-repo=$HOME/repo
   done; done
   aws s3 sync results/raw s3://<BUCKET>/sp/results/raw/
   ```

11. On the g7e: OpenFold-TRT, the AF2-class reference (BF16 Evoformer, TRT 10.13 / CUDA 13.0,
   model_1 weights, one unrelaxed structure). Install OpenFold at the pinned commit following
   its docs, build the TensorRT engine once, then run seeds 0 and 1. Alignments: one directory
   per target holding that target's shared `.a3m` file; templates are not used.

   ```bash
   git clone https://github.com/aqlaboratory/openfold.git ~/openfold
   git -C ~/openfold checkout be2ec1841f16c966c65ae0e7599ebbadc725757d
   # install per ~/openfold/docs/source/Installation.md, then download params_model_1.npz
   mkdir -p ~/of/fasta ~/of/aln ~/of/empty_templates
   python - <<'EOF'
   import json, pathlib
   for t in json.load(open("targets/targets.json"))["targets"]:
       if t["set"] != "casp14": continue
       ch = t["chains"][0]
       pathlib.Path(f"/home/ubuntu/of/fasta/{t['id']}").mkdir(parents=True, exist_ok=True)
       pathlib.Path(f"/home/ubuntu/of/fasta/{t['id']}/{t['id']}.fasta").write_text(f">{t['id']}\n{ch['sequence']}\n")
       d = pathlib.Path(f"/home/ubuntu/of/aln/{t['id']}"); d.mkdir(parents=True, exist_ok=True)
       (d / "colabfold.a3m").write_bytes(pathlib.Path(f"/data/msas/{ch['msa_id']}.a3m").read_bytes())
   EOF
   # one-time engine build (timings discarded): same command with --trt_mode build
   for t in $(ls ~/of/fasta); do for s in 0 1; do
     PYTHONPATH=. python -m harness.run predict --backend openfold-trt --target-id $t \
       --input ~/of/fasta/$t/$t.fasta --out-dir results/raw/g7e-openfold/$t/seed$s --seed $s \
       --alignment-dir ~/of/aln --template-mmcif-dir ~/of/empty_templates \
       --jax-param-path ~/openfold/openfold/resources/params/params_model_1.npz \
       --trt-engine-dir ~/of/trt --openfold-dir ~/openfold --targets-json targets/targets.json
   done; done
   aws s3 sync results/raw/g7e-openfold s3://<BUCKET>/sp/results/raw/g7e-openfold/
   ```

   The Boltz-2 NIM column is supplementary and off by default; run it only if licensing permits,
   and record that decision on #72.

## Days 3–7 — trn2 runs (Phase A tasks 3, 6, 7; #73, #76, #77)

12. Launch the trn2.3xlarge as in RUNBOOK-A0 steps 2–6 (setup and pinned boltz-neuron). Record
    boot and setup, pull the MSAs and write the inputs.

    ```bash
    cd ~/repo/benchmarks/structure-prediction && PYTHONPATH=. python -m harness.stages boot --out ~/stages.json --role trn2
    # (RUNBOOK-A0 step 5's setup.sh, run as: python -m harness.stages time setup --out ~/stages.json -- bash ~/boltz-neuron/scripts/setup.sh)
    aws s3 sync s3://<BUCKET>/sp/msas/ ~/msas/
    cd ~/repo/benchmarks/structure-prediction
    PYTHONPATH=. python -m harness.run inputs --targets targets/targets.json --msa-dir ~/msas --out ~/inputs
    ```

13. Build the NEFF cache for the CASP14 shape set at each LNC (minutes per shape; the host has
    124 GB RAM, enough for one compile at a time). Publish each cache; it is a Phase A
    deliverable.

    ```bash
    for lnc in 2 1; do
      NEURON_LOGICAL_NC_CONFIG=$lnc PYTHONPATH=. python -m harness.stages time cache_build_lnc$lnc --out ~/stages.json -- \
        bash ~/boltz-neuron/scripts/build_cache.sh ~/inputs 1,5
      mv ~/boltz_neuron_cache.tgz ~/sp_cache_sdk233_lnc${lnc}_casp14_ds1-5.tgz
      aws s3 cp ~/sp_cache_sdk233_lnc${lnc}_casp14_ds1-5.tgz s3://<BUCKET>/sp/neff-cache/
    done
    ```

14. Comparison runs: LNC=2 with 4 workers and LNC=1 with 8, ds=1 and ds=5, seeds 0 and 1, with
    a profiled extra run per input (neuron-profile via `NEURON_RT_INSPECT_*`).

    ```bash
    for lnc in 2 1; do w=$([ $lnc = 2 ] && echo 4 || echo 8); for ds in 1 5; do for s in 0 1; do
      PYTHONPATH=. python -m harness.run batch --backend neuron --lnc $lnc --workers $w --ds $ds \
        --seed $s --inputs ~/inputs --out-root results/raw/trn2 --profile \
        --targets-json targets/targets.json --repo boltz-neuron=$HOME/boltz-neuron --repo this-repo=$HOME/repo
    done; done; done
    aws s3 sync results/raw/trn2 s3://<BUCKET>/sp/results/raw/trn2/
    ```

    Inputs above the per-core ceiling (the two complexes, and T1061 if it does not fit at the
    default MSA depth) are recorded as `did-not-run`. That is the ceiling column, not an error,
    and the primary setting is not changed to make them fit.

15. The T1061 secondary row (`--max_msa_seqs 2048`) at both LNC settings, ds=1 and ds=5. Then the
    end-to-end batch runs: restore the published cache as a timed stage (what a deployment
    would do), then the 20-target directory per LNC layout.

    ```bash
    for lnc in 2 1; do for ds in 1 5; do for s in 0 1; do
      NEURON_LOGICAL_NC_CONFIG=$lnc PYTHONPATH=. python -m harness.run predict --backend neuron --lnc $lnc --core 0 \
        --target-id T1061 --input ~/inputs/T1061.yaml --out-dir results/raw/trn2-msa2048/lnc$lnc-ds$ds/T1061/seed$s \
        --seed $s --ds $ds --variant msa2048 --max-msa-seqs 2048 --targets-json targets/targets.json
    done; done; done
    mkdir -p ~/inputs-casp && for t in $(python -c "import json;print(' '.join(t['id'] for t in json.load(open('targets/targets.json'))['targets'] if t['set']=='casp14'))"); do cp ~/inputs/$t.yaml ~/inputs-casp/; done
    for lnc in 2 1; do w=$([ $lnc = 2 ] && echo 4 || echo 8)
      rm -rf /tmp/neff_cache
      PYTHONPATH=. python -m harness.stages time cache_restore_lnc$lnc --out ~/stages.json -- \
        bash ~/boltz-neuron/scripts/restore_cache.sh ~/sp_cache_sdk233_lnc${lnc}_casp14_ds1-5.tgz
      for ds in 1 5; do for s in 0 1; do
        NEURON_LOGICAL_NC_CONFIG=$lnc PYTHONPATH=. python -m harness.run directory --backend neuron --lnc $lnc --workers $w \
          --inputs ~/inputs-casp --out-dir results/raw/trn2-dir/lnc$lnc-ds$ds-s$s --seed $s --ds $ds --repo boltz-neuron=$HOME/boltz-neuron
      done; done
    done
    aws s3 sync results/raw s3://<BUCKET>/sp/results/raw/
    aws s3 cp ~/stages.json s3://<BUCKET>/sp/stages/trn2.json
    ```

    Each LNC layout bills its own cache restore (`cache_restore_lnc2` / `cache_restore_lnc1`
    in the one `stages.json`).

16. Atom-graph split check (task 6): the two largest CASP14 targets at LNC=2 with and without
    the split, seeds 0 and 1, ds=1.

    ```bash
    for t in T1061 T1094; do for agb in 0 1; do for s in 0 1; do
      PYTHONPATH=. python -m harness.run predict --backend neuron --lnc 2 --core 0 --atom-graph-break $agb \
        --target-id $t --input ~/inputs/$t.yaml --out-dir results/raw/trn2-agb/agb$agb/$t/seed$s \
        --seed $s --ds 1 --targets-json targets/targets.json --repo boltz-neuron=$HOME/boltz-neuron
    done; done; done
    aws s3 sync results/raw/trn2-agb s3://<BUCKET>/sp/results/raw/trn2-agb/
    ```

17. Tear down the trn2 and g7e hosts once their results and `stages.json` are in S3 (on the g7e:
    `aws s3 cp ~/stages.json s3://<BUCKET>/sp/stages/g7e.json`).

    ```bash
    python infra/launch.py teardown trn2 --yes
    python infra/launch.py teardown g7e --yes
    ```

## After the oracle finishes — scoring, profiles, cost table (tasks 4, 5; #74, #75)

18. On the workstation, pull everything and score (top-ranked sample gated, best-of-N reported).

    ```bash
    aws s3 sync s3://<BUCKET>/sp/results/ results/raw-all/
    aws s3 sync s3://<BUCKET>/sp/timing/ results/timing/
    python -m harness.score score --results results/raw-all --targets targets/targets.json \
      --truth-dir ~/sp-data/truth --out results/scores_phaseA.csv
    ```

19. Gates, in this order (the ds=5 gates refuse to run until the g7e ds=1 gate has passed).

    ```bash
    S="--scores results/scores_phaseA.csv --targets targets/targets.json"
    python -m harness.score gate $S --acc backend=gpu,ds=1    --ref backend=cpu,ds=1 --out results/gate_g7e_ds1
    python -m harness.score gate $S --acc backend=neuron,ds=1,lnc=2,atom_graph_break= --ref backend=cpu,ds=1 --out results/gate_trn2_lnc2_ds1
    python -m harness.score gate $S --acc backend=neuron,ds=1,lnc=1,atom_graph_break= --ref backend=cpu,ds=1 --out results/gate_trn2_lnc1_ds1
    python -m harness.score gate $S --acc backend=neuron,ds=5,lnc=2,atom_graph_break= --ref backend=gpu,ds=5 \
      --prereq results/gate_g7e_ds1.json --out results/gate_trn2_lnc2_ds5
    python -m harness.score gate $S --acc backend=neuron,ds=5,lnc=1,atom_graph_break= --ref backend=gpu,ds=5 \
      --prereq results/gate_g7e_ds1.json --out results/gate_trn2_lnc1_ds5
    python -m harness.score gate $S --acc backend=neuron,ds=1,lnc=2,atom_graph_break=0 --ref backend=cpu,ds=1 --out results/gate_trn2_agb0
    python -m harness.score gate $S --acc backend=neuron,ds=1,lnc=2,atom_graph_break=1 --ref backend=cpu,ds=1 --out results/gate_trn2_agb1
    # secondary row (T1061, --max_msa_seqs 2048): per-target, no phase verdict
    python -m harness.score gate $S --acc backend=gpu,ds=1,variant=msa2048 --ref backend=cpu,ds=1,variant=msa2048 --out results/gate_g7e_msa2048_ds1
    python -m harness.score gate $S --acc backend=neuron,ds=1,lnc=2,variant=msa2048 --ref backend=cpu,ds=1,variant=msa2048 --out results/gate_trn2_lnc2_msa2048_ds1
    python -m harness.score gate $S --acc backend=neuron,ds=1,lnc=1,variant=msa2048 --ref backend=cpu,ds=1,variant=msa2048 --out results/gate_trn2_lnc1_msa2048_ds1
    ```

    The primary gates cover the 19 targets with ground truth and need at least 17 to pass. T1047s1
    is listed as "no ground truth, timing only".

    The last two (task 6) cover only T1061 and T1094, so their phase verdict is FAIL by
    construction; read the two targets' rows in the `.md` files. The defect is present on the SDK
    in use if `agb0` fails a target that `agb1` passes. Comment the result on #76.

20. Classify the profiles (per run; aggregate per platform for the occupancy breakdown).

    ```bash
    for p in $(find results/raw-all -name torch-profile.json); do
      python -m harness.profile_classify gpu --trace $p --out ${p%.json}.classes.json; done
    for d in $(find results/raw-all -type d -name neuron-profile); do
      # export each device profile to JSON first: neuron-profile view --output-format json ...
      for j in $d/*.json; do python -m harness.profile_classify neuron --profile $j --out ${j%.json}.classes.json; done; done
    ```

    The Neuron JSON field mapping (`NEURON_FIELDS` in `harness/profile_classify.py`) has not been
    checked against real `neuron-profile` output yet. Check it on the first profile and pass
    `--fields` if it differs; comment the mapping used on #74.

21. Build the cost table. Prices come from the run manifests, MSA terms from the timing files,
    and end-to-end batch billing from the stage files.

    ```bash
    aws s3 sync s3://<BUCKET>/sp/stages/ results/stages/
    python -m harness.cost_table --results results/raw-all --targets targets/targets.json \
      --msa-gpu results/timing/msa_gpu_timing.json --msa-cpu results/timing/msa_cpu_timing.json \
      --stages results/stages/*.json --out results/phaseA.csv --markdown results/phaseA.md
    ```

    - **Primary:** one top-ranked prediction per target ("DL warm", which excludes start-up).
    - **Secondary:** per diffusion sample.
    - **End-to-end batch** (on the mean rows): billed boot, model load, MSA and cache restore,
      amortised over the 20-target directory run. Setup stages are reported but billed only with
      `--bill-setup`.

22. Tear down the c7a and confirm nothing is left running.

    ```bash
    python infra/launch.py teardown c7a --yes
    python infra/launch.py list
    ```

23. Commit `results/phaseA.csv`, `results/phaseA.md`, the gate `.json`/`.md` files, the scores
    CSV and the run manifests (not predictions or profiles); open a PR that references #55.
    Post the one-page Phase A note as a comment on #55, scoring each item of
    `results/EXPECTATIONS.md` against the results, and the gate outcome that sets the Phase B
    order. The owner decides when Phase A is complete.
