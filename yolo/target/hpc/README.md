```bash

# connect to biihead
# if you dont, then the apptainer build is Killed,
# the cms ee gets Killed...
#
module load apptainer
apptainer build --force my_darknet_container.sif apptainer.def

source ~/ENV3/bin/activate
pip install git+https://github.com/cloudmesh/cloudmesh-ee.git

cms ee generate \
  --source=slurm.in.sh \
  --config=config.yaml \
  --name=chocolatechip_runs \
  --output_dir=project \
  --mode=h


cms ee generate submit --name=chocolatechip_runs --job_type=slurm > submit.sh
bash submit.sh
```


# ultralytics

```bash
apptainer build --force my_ultralytics_container.sif apptainer.ultra.def

source ~/ENV3/bin/activate
pip install git+https://github.com/cloudmesh/cloudmesh-ee.git

cms ee generate \
  --source=slurmultra.in.sh \
  --config=config.yaml \
  --name=chocolatechip_runs \
  --output_dir=project \
  --mode=h

cms ee generate submit --name=chocolatechip_runs --job_type=slurm > submit.sh
bash submit.sh
```


