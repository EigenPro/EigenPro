# Get the directory path of the script
script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
root_dir="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
export PYTHONPATH=${root_dir}:$PYTHONPATH

python ${script_dir}/run_example.py --dataset=fmnist
