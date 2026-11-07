<!-- Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0 -->

# RUNBOOK-A0: reproduce the port's published trn2 number

Phase A task 0 (#70): before any comparison run, reproduce boltz-neuron's published warm latency
on trn2.3xlarge: the 8eil example (186 residues, padded to 192 tokens), ds=1, 50 sampling steps,
LNC=2, prebuilt cache. Expected about 7.5 s. **A mismatch of more than 10 % stops Phase A** until
it is explained (SDK drift, cache mismatch, instance variant). Record the outcome as a comment on
#70.

The 50-step number is a reproduction target only, never a comparison row.

The port's method (its README): warm model time per prediction, the median of 3 after 1 warm-up,
in one predict process. Step 8 below does the same: one process over four copies of the input,
timed per prediction by `harness.entry`.

## On your workstation

1. Clone this repository and enter the harness directory.

   ```bash
   git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research.git
   cd aws-trainium-tutorial-for-research/benchmarks/structure-prediction
   ```

2. Pick a region and AZ that offer trn2.3xlarge. The trn2 host's `region` is in
   `infra/hosts.yaml` (sa-east-1); `--region` overrides it per command.

   ```bash
   aws ec2 describe-instance-type-offerings --location-type availability-zone \
     --filters Name=instance-type,Values=trn2.3xlarge --region us-east-1 \
     --query 'InstanceTypeOfferings[].Location' --output text
   ```

3. Review the launch plan (trn2 uses the `aws ec2 run-instances` fallback until
   spore-host/spawn#205 lands). It resolves the SDK 2.33 DLAMI from SSM and changes nothing.

   ```bash
   python infra/launch.py plan trn2 --az <AZ> --key-name <KEY>
   ```

4. Launch. The instance terminates itself after its TTL (24 h). Inside a purchased Capacity
   Block (trn2.3xlarge blocks are offered only in sa-east-1 and ap-southeast-4), pass the
   block's reservation ID and use its region and AZ. `hosts.yaml` defaults trn2 to sa-east-1,
   and `--region` overrides it.

   ```bash
   python infra/launch.py launch trn2 --az <AZ> --key-name <KEY> --yes
   # or, in a Capacity Block:
   python infra/launch.py launch trn2 --region <BLOCK_REGION> --az <BLOCK_AZ> --reservation-id <cr-...> --key-name <KEY> --yes
   python infra/launch.py list        # note the instance ID and connect with SSH or SSM
   ```

## On the trn2.3xlarge

5. Clone this repository and boltz-neuron at the pinned commit, then install Boltz-2 into the
   DLAMI venv with the port's setup script.

   ```bash
   git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research.git ~/repo
   git clone https://github.com/jimburtoft/boltz-neuron.git ~/boltz-neuron
   git -C ~/boltz-neuron checkout 7cf012186a08cff5fff0d6718049d4e73df8b8d9
   cd ~/boltz-neuron && bash scripts/setup.sh
   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
   ```

6. Restore the port's prebuilt LNC=2 cache.

   ```bash
   CACHE=boltz_neuron_cache_v9_sdk233_trn2_lnc2_ds1-8-25.tgz
   hf download jburtoft/boltz-neuron-cache $CACHE --repo-type dataset --local-dir ~/
   bash ~/boltz-neuron/scripts/restore_cache.sh ~/$CACHE
   ```

7. Make four copies of the example (one warm-up, three timed), under distinct names.

   ```bash
   mkdir -p ~/a0/inputs
   for i in 1 2 3 4; do cp ~/boltz-neuron/examples/8eil.yaml ~/a0/inputs/8eil_$i.yaml; done
   ```

8. Run one predict process over the four inputs, timed per prediction, LNC=2 on logical core 0.

   ```bash
   cd ~/repo/benchmarks/structure-prediction
   export NEURON_RT_VISIBLE_CORES=0
   PYTHONPATH=. python -m harness.entry --target boltz_neuron --timing-out ~/a0/timing.json -- \
     predict ~/a0/inputs --out_dir ~/a0/out --recycling_steps 3 --sampling_steps 50 \
     --diffusion_samples 1 --num_workers 0 --seed 0 --override
   ```

9. Capture the run manifest (versions, commits, LNC, environment, prices fetched now).

   ```bash
   PYTHONPATH=. python -m harness.manifest --accelerator neuron --lnc 2 \
     --repo boltz-neuron=$HOME/boltz-neuron --repo this-repo=$HOME/repo --out ~/a0/manifest.json
   ```

10. Apply the 10 % stop rule. Exit code 0 is within tolerance; 3 means stop.

    ```bash
    PYTHONPATH=. python -m harness.repro_check --timing ~/a0/timing.json --expect 7.5 --tol 0.10
    ```

11. Post the JSON line from step 10, `~/a0/timing.json` and the manifest's version block as a
    comment on #70. If it is outside tolerance, add `status:blocked` to #70 and name the
    suspected cause. Do not continue to RUNBOOK-A.

## Teardown

12. From the workstation:

    ```bash
    python infra/launch.py teardown trn2 --yes
    python infra/launch.py list        # expect no trn2 instance
    ```
