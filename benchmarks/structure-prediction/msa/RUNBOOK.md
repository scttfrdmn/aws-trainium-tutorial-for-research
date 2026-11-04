<!-- Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0 -->

# MSA stage runbook

One set of MSAs, generated once on the g7e with MMseqs2-GPU and used unchanged by every
platform (CPU oracle, g7e, trn2, OpenFold-TRT). The trn2 pipeline's end-to-end MSA term is a
separate timing of MMseqs2-CPU on an r7a.48xlarge; its output is discarded.

## Versions

| component | version |
|---|---|
| ColabFold (`colabfold_search`, `setup_databases.sh`) | v1.6.3 |
| MMseqs2 | 18-8cc5c (`8cc5ce367b5638c4306c2d7cfc652dd099a4643f`); GPU build on g7e, AVX2 build on r7a |
| UniRef30 | `uniref30_2302` (2023_02) |
| ColabFoldDB | `colabfold_envdb_202108` (2021_08) |
| Templates | not used (`SKIP_TEMPLATES`; Boltz-2 runs MSA-only by default, ColabFold's default is no templates) |

`stage_databases.sh` writes `DATABASES.txt` (setup tag, `mmseqs version`, ready markers, size) next
to the databases; `run_search.py` records the versions and per-target wall time in the timing
JSON with a run manifest (instance, prices).

## Storage and memory

* The prebuilt databases are downloaded once (`DOWNLOADS_ONLY=1`) and staged through S3, then
  indexed on each host: `GPU=1` (padded, GPU index) on the g7e, the default CPU index on the r7a.
  Both indexes are built from the same downloaded files, so the database contents are the same.
  ColabFold notes that MMseqs2-CPU results on these prebuilt databases can differ from GPU results
  because of database ordering; that does not matter here, because the CPU output is discarded and
  only its time is used.
* Indexed size is about 1 TB; each host gets a 2 TB gp3 data volume at `/data`.
* Host RAM (ColabFold README, v1.6.3): batch searches without a precomputed index need about
  **128 GB**; keeping the index resident for fast single-query search needs **768 GB to 1 TB**
  for ColabFoldDB. trn2.3xlarge has 124 GB, so a CPU search there would measure disk paging; it
  is not used. The r7a.48xlarge (1,536 GiB) holds the index in RAM (`vmtouch`, then
  `--db-load-mode 2`).

## Steps

1. Stage (once, any host with ~1 TB free, e.g. the g7e):
   `bash msa/stage_databases.sh upload /data/colabfold s3://<BUCKET>/sp/colabfold-dbs`
2. g7e: `bash msa/stage_databases.sh restore /data/colabfold s3://<BUCKET>/sp/colabfold-dbs gpu`
3. g7e: `bash msa/msa_gpu.sh /data/colabfold targets/targets.json /data/msas /data/msa_gpu_timing.json`
   (starts `mmseqs gpuserver` for both databases, then one `colabfold_search --gpu 1 --gpu-server 1
   --db-load-mode 2` per unique protein chain; writes `<msa_id>.a3m`).
4. Publish: `aws s3 sync /data/msas s3://<BUCKET>/sp/msas/` and copy the timing JSON + manifest.
5. r7a: `bash msa/stage_databases.sh restore /data/colabfold s3://<BUCKET>/sp/colabfold-dbs cpu`
6. r7a: `bash msa/msa_cpu_timing.sh /data/colabfold targets/targets.json /data/msa_cpu_timing.json`
   (refuses to run below 768 GiB RAM; preloads the index; output discarded).
7. Publish the CPU timing JSON + manifest; tear the r7a down.

Never regenerate MSAs per platform. If an `.a3m` has to change, every platform's results that
used it are rerun, and the change is recorded on #61.
