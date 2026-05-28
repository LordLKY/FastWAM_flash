cd /home/kyliu/Robotics/code/FastWAM_flash_dev

source ~/miniconda3/etc/profile.d/conda.sh

conda activate fastwam

export PYTHONPATH=/home/kyliu/Robotics/data/LIBERO:$PYTHONPATH

# python experiments/libero/run_libero_manager.py task=libero_cache_2cam224_1e-4 ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1 EVALUATION.num_trials=1

# python experiments/libero/run_libero_manager.py task=libero_cache_2cam224_1e-4 ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1 MULTIRUN.task_suite_names="[libero_object]"

python experiments/libero/run_libero_manager.py task=libero_cache_2cam224_1e-4 ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1 EVALUATION.record_videos=false EVALUATION.adaptive_horizon=true MULTIRUN.task_suite_names="[libero_spatial]"  #  EVALUATION.num_trials=10  # MULTIRUN.task_suite_names="[libero_goal, libero_object, libero_spatial]" # EVALUATION.record_actions=true EVALUATION.absolute_actions=true EVALUATION.replan_steps=5  # EVALUATION.record_actions_bias=true