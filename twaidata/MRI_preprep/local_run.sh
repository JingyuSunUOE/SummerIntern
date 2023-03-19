# script to call the preprocessing code
# in the pgr cluster for the wmh challenge and the ed inhouse datasets

# argument 1 is which individual to start at, argument 2 is which individual to stop at

# paths to datasets on machine
in_dir_wmhchal=/Users/sunjingyu/Downloads/whole_data/preprocessed_data
in_dir_ed=/home/s2208943/ipdis/data/core_data
out_dir=/Users/sunjingyu/Downloads/whole_data/final_data

# process WMH dataset
python simple_preprocess_two.py -i ${in_dir_wmhchal} -o ${out_dir} -n WMH_challenge_dataset -s $1 -e $2 -f $3

# process ed inhouse dataset
# but ed data is on clusyer only, but this is what i would run.
# python simple_preprocess.py -i ${in_dir_ed} -o ${out_dir} -n WMH_challenge_dataset -s "0" -e "-1"