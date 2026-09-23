# DLG experiments
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_dlg.yaml --gpu-ids 7 --occupy-after-run --occupancy-script /home/zx/nas/gpu/train_stealth.py --execute

# Inverting Gradients experiments
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_ig.yaml --gpu-ids 7 --occupy-after-run --occupancy-script /home/zx/nas/gpu/train_stealth.py --execute

# Inverting Gradients image-only correctness experiment
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_ig_image_only.yaml --gpu-ids 7 --occupy-after-run --occupancy-script /home/zx/nas/gpu/train_stealth.py --execute

# Inverting Gradients image-only learning-rate sweep (adaptive stage 1; no occupancy)
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_ig_lr_sweep.yaml --gpu-ids 7 --execute

# Inverting Gradients image-only lower-bound learning-rate test (50 GiB threshold)
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_ig_lr_lower_sweep.yaml --gpu-ids 7 --min-free-mib 50000 --execute

# Inverting Gradients image-only TV sweep at the selected image learning rate
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_ig_tv_sweep.yaml --gpu-ids 7 --min-free-mib 50000 --execute

# Inverting Gradients image-only iteration sweep at the selected lr and tv
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_ig_iteration_sweep.yaml --gpu-ids 7 --min-free-mib 50000 --execute

# Inverting Gradients image-only iteration sweep occ at the selected lr and tv
conda run --no-capture-output -n gia python -m utils.run_cmds --cmd-config-yaml run_yaml/slake_llava_ig_iteration_occ.yaml --gpu-ids 6 --min-free-mib 50000 --execute