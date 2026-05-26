# Resume the cartoon LoRA from step-1200. The prior run was interrupted around
# step 1369. We restart from the last saved checkpoint and continue training.
#
# Caveats:
# - AdamW optimizer state is NOT restored (LoRA-fine: usually negligible).
# - The new train.log restarts the step counter from 1; treat new step N as
#   global step 1200 + N.
# - Output dir is new (_resume_from_1200) so the old step-200..step-1200 files
#   under .../lora_cartoon/ stay intact.
#
# Run from DiffSynth-Studio repo root.
# Override GPU count with:  NUM_GPUS=4 bash <this-script>

NUM_GPUS=${NUM_GPUS:-2}

accelerate launch \
  --num_processes=$NUM_GPUS \
  --mixed_precision=bf16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed \
  --dataset_metadata_path ./data/cartoon_diffsynth/metadata.csv \
  --data_file_keys "video,control_video" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_resume_from_1200" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video" \
  --use_gradient_checkpointing \
  --lora_checkpoint "./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon/step-1200.safetensors"
