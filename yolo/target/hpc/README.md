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
# destination for yolov7 benchmarks
DST_ROOT="/blue/ranka/j.fleischer/tempdele/chocolatechip/semester-work/spring2025/darknet/artifacts/outputs/FisheyeTrafficDarknetLocal"
mkdir -p "${DST_ROOT}/yolov7"

# move all benchmark__* folders from every repeat’s yolov7 subdir
for d in directive_*_repeat_*; do
  if [ -d "${d}/yolov7" ]; then
    b=$(find "${d}/yolov7" -maxdepth 1 -type d -name 'benchmark__*' -print -quit)
    [ -n "$b" ] && mv "$b" "${DST_ROOT}/yolov7/" && echo "Moved $(basename "$b") → yolov7"
  fi
done

for d in directive_*_repeat_*; do
  if { [ ! -d "$d/yolov7" ] || [ -z "$(find "$d/yolov7" -mindepth 1 -print -quit 2>/dev/null)" ]; } && \
     { [ ! -d "$d/outputs" ] || [ -z "$(find "$d/outputs" -type f -print -quit 2>/dev/null)" ]; }; then
    rm -rf -- "$d"
    echo "Deleted $d"
  fi
done
```
