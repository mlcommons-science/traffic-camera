```bash

# connect to biihead
# if you dont, then the apptainer build is Killed,
# the cms ee gets Killed...
#
module load apptainer
apptainer build --force my_darknet_container.sif apptainer.def

source ~/ENV3/bin/activate

pip uninstall cloudmesh-ee
pip uninstall cloudmesh-rivanna
pip install git+https://github.com/cloudmesh/cloudmesh-rivanna.git -U
pip install git+https://github.com/cloudmesh/cloudmesh-ee.git -U

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

pip uninstall cloudmesh-ee
pip uninstall cloudmesh-rivanna
pip install git+https://github.com/cloudmesh/cloudmesh-rivanna.git -U
pip install git+https://github.com/cloudmesh/cloudmesh-ee.git -U

cms ee generate \
  --source=slurmultra.in.sh \
  --config=config.yaml \
  --name=chocolatechip_runs \
  --output_dir=project \
  --mode=h

cms ee generate submit --name=chocolatechip_runs --job_type=slurm > submit.sh
bash submit.sh
```

# hipergator

do the same commands, but replace config.yaml with config.ufl.yaml

# move runs


```bash
DST_ROOT="/blue/ranka/j.fleischer/tempdele/chocolatechip/semester-work/spring2025/darknet/artifacts/outputs/FisheyeTrafficDarknetLocal"
mkdir -p "${DST_ROOT}"

for d in directive_*_repeat_*; do
  ydir=$(find "$d" -maxdepth 1 -type d -name 'yolov*' -printf '%f\n' | head -n 1)
  [ -z "$ydir" ] && continue

  mkdir -p "${DST_ROOT}/${ydir}"

  b=$(find "${d}/${ydir}" -maxdepth 1 -type d -name 'benchmark__*' -print -quit)
  [ -n "$b" ] && mv "$b" "${DST_ROOT}/${ydir}/" && echo "Moved $(basename "$b") → ${ydir}"
done

for d in directive_*_repeat_*; do
  ydir=$(find "$d" -maxdepth 1 -type d -name 'yolov*' -printf '%f\n' | head -n 1)

  if { [ -z "$ydir" ] || [ -z "$(find "$d/$ydir" -mindepth 1 -print -quit 2>/dev/null)" ]; } && \
     { [ ! -d "$d/outputs" ] || [ -z "$(find "$d/outputs" -type f -print -quit 2>/dev/null)" ]; }; then
    rm -rf -- "$d"
    echo "Deleted $d"
  fi
done
```
