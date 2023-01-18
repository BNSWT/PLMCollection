CUDA_VISIBLE_DEVICES=4 python3 Evoformer/run_pretrained_openfold.py \
        output/test-alignments \
        /home/public/OpenFold/dataset/pdb_mmcif/mmcif_files/ \
        --use_precomputed_alignments output/test-alignments/alignments\
        --uniref90_database_path  /home/public/OpenFold/dataset/uniref90/uniref90.fasta \
        --pdb70_database_path  /home/public/OpenFold/dataset/pdb70/pdb70 \
        --uniclust30_database_path  /home/public/OpenFold/dataset/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
        --output_dir ./output/test-alignments/pdb/ \
        --model_device 'cuda:0' \
        --hhsearch_binary_path ./lib/conda/envs/plms/bin/hhsearch \
        --kalign_binary_path ./lib/conda/envs/plms/bin/kalign \
        --config_preset 'model_1_ptm' \
        --openfold_checkpoint_path /home/public/OpenFold/resources/openfold_params/finetuning_ptm_2.pt \
        --skip_relaxation
