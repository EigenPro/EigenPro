# Get the directory path of the script
script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

python ${script_dir}/run_example.py --dataset=fmnist
