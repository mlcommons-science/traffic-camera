```bash
cms ee generate slurm.in.sh \
  --config=config.yaml \
  --name=chocolatechip_runs \
  --output_dir=project \
  --mode=h \
  --verbose

cms ee generate submit --name=chocolatechip_runs --job_type=slurm
```
