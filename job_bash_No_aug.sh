#PBS -N supervised_MLC_no_aug_3c_balanced
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=1:ubuntu2004:nvidiaGTX1080Ti,mem=8gb,walltime=24:00:00
#PBS -j oe
#PBS -o /misc/student/alzouabk/Thesis/supervised_multi_label_classification/outputs/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate Thesis_CT_scans

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/supervised_multi_label_classification/src/train_aug_exp.py -cp '/misc/student/alzouabk/Thesis/supervised_multi_label_classification/configs/config_No_Aug.yml'
