# LoRA smoke test on humanvid-subset1 (150 samples) for Wan2.1-Fun-V1.1-1.3B-Control.
# Goal: verify pipeline + see loss decrease, then scale up.
# Trains on (video, pose) only — no reference_image. V1.1 base handles
# reference_image=None via CFG dropout, so inference can still use chain ref.
#
# Run from DiffSynth-Studio repo root.
# Override GPU count with:  NUM_GPUS=4 bash <this-script>

NUM_GPUS=${NUM_GPUS:-2}

accelerate launch \
  --num_processes=$NUM_GPUS \
  --mixed_precision=bf16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed \
  --dataset_metadata_path ./data/humanvid_diffsynth/metadata.csv \
  --data_file_keys "video,control_video" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_humanvid_smoke" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video" \
  --use_gradient_checkpointing
