"""
Zero-shot chunk-chaining sanity check for the LTX-2 sink/recent mapping.

Validates, BEFORE any training, that the conditioning mapping chosen for the
recycle port actually chains on the base model:
    recent → input_images index 0 (frame-0 latent replacement)
    sink   → input_images index --sink_index (t=0 reference token; negative
             index = past-time position — the thing this script must verify
             RoPE tolerates)

Chain convention mirrors the Wan few-step script
(Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py): sink fixed across
chunks, recent = previous chunk's last frame, 1-frame overlap (chunk k>0
drops its frame 0 at stitch time).

What to look at in the output:
  - Does chunk k>0 actually continue from the recent (no scene reset)?
  - Identity/style drift across chunks WITHOUT recycle — this is the
    baseline degradation that recycle training is supposed to fix; if the
    base model shows zero drift over many chunks there is nothing to fix.
  - Garbage/artifacts when sink token is enabled vs --no_sink → tells us
    whether the negative-index position is usable. Try --sink_index -49,
    -8, and --no_sink.

Usage (single GPU):
    python examples/ltx2/model_inference/LTX-2-Chunk-Chain-Sanity.py \
        --initial_ref path/to/first_frame.png --prompt "..." \
        --num_chunks 4 --sink_index -49
"""
import argparse, os
import torch
from PIL import Image

from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="[VISUAL]:A cartoon girl dancing in a bright room, anime style, high quality. [SOUNDS]:soft background music")
parser.add_argument("--negative_prompt", default="blurry, out of focus, low contrast, washed out colors, excessive noise, deformed facial features, extra limbs, disfigured hands")
parser.add_argument("--initial_ref", default=None,
                    help="First-frame image. If omitted, chunk 0 is generated "
                         "from the prompt alone and its frame 0 becomes the sink.")
parser.add_argument("--num_chunks", type=int, default=4)
parser.add_argument("--chunk_frames", type=int, default=49, help="Must be ≡ 1 mod 8.")
parser.add_argument("--height", type=int, default=832)
parser.add_argument("--width", type=int, default=480)
parser.add_argument("--frame_rate", type=int, default=24)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--cfg", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sink_index", type=int, default=-49,
                    help="Reference-token index for the sink (≠ 0).")
parser.add_argument("--no_sink", action="store_true",
                    help="Recent-only conditioning (ablate the sink token).")
parser.add_argument("--input_images_strength", type=float, default=1.0)
parser.add_argument("--lora_path", default=None,
                    help="Trained LoRA(s), comma-separated, loaded in order — "
                         "e.g. Stage-A ckpt for teacher eval, or "
                         "'stageA.safetensors,studentB.safetensors' for the "
                         "1-step student chain (pair with --steps 1).")
parser.add_argument("--lora_alpha", type=float, default=1.0)
parser.add_argument("--control_video", default=None,
                    help="Optional pose/control video driving motion via the "
                         "IC-LoRA in_context path. Sliced per chunk (looped "
                         "if shorter than the chain) — gives cross-chunk "
                         "velocity continuity like Wan's Fun-Control.")
parser.add_argument("--ic_lora", default=("models/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/"
                                          "ltx-2-19b-ic-lora-pose-control.safetensors"),
                    help="IC-LoRA weights as 'model_id:pattern' (downloaded) "
                         "or a local .safetensors path. Fused when "
                         "--control_video is given.")
parser.add_argument("--ic_downsample", type=int, default=2)
parser.add_argument("--time_shift_frames", type=int, default=0,
                    help="Positive-shift sink mapping: chunk time coords +N "
                         "frames, sink at small POSITIVE index (~t=0). Use 48 "
                         "with --sink_index 1. MUST match training config.")
parser.add_argument("--base", choices=["19b", "2.3"], default="19b",
                    help="Backbone: 19b = local ltx-2-19b-dev (our main line); "
                         "2.3 = ltx-2.3-22b-dev (for evaluating the 2.3 union "
                         "IC-LoRA — pass the matching --ic_lora!).")
parser.add_argument("--output", default="ltx2_chain_sanity.mp4")
args = parser.parse_args()
assert args.chunk_frames % 8 == 1
assert args.sink_index != 0

BASE_CONFIGS = {
    "19b": dict(path="/mnt/vita/scratch/vita-students/users/jinghao/code/LTX-2-old/models/LTX-2/ltx-2-19b-dev.safetensors"),
    "2.3": dict(path="models/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors"),
}

# offload_device=cuda → models stay RESIDENT on GPU (no per-chunk CPU↔GPU
# shuttling). 22B DiT (~44GB) + gemma-12B (~24GB) + VAEs ≈ 70GB, fits H200
# 140GB for 1-step inference. If it OOMs, revert offload_device to "cpu".
vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cuda",
    "onload_dtype": torch.bfloat16, "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
        # Official combined checkpoint: DiffSynth auto-loads ALL submodules
        # from it by hash (dit, video/audio VAEs, vocoder,
        # text_encoder_post_modules) — no Repackage download needed.
        ModelConfig(**BASE_CONFIGS[args.base], **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
)
if args.lora_path:
    for lp in args.lora_path.split(","):
        pipe.load_lora(pipe.dit, lora_config=lp.strip(), alpha=args.lora_alpha)
        print(f"[lora] loaded {lp.strip()} (alpha={args.lora_alpha})")

if args.steps == 1:
    # set_timesteps_ltx2's terminal shift divides 0/0 on a single-point grid
    # (sigma grid = [1.0] → NaN). 1-NFE denoises from pure noise at t=1000;
    # Euler step to sigma_=0 then yields x0 = x_t - v exactly. Same special
    # case as derive_denoising_step_list in dmd_utils_ltx2.py.
    import types
    def _one_step_schedule(self_, num_inference_steps=1, denoising_strength=1.0, **kw):
        self_.sigmas = torch.tensor([1.0])
        self_.timesteps = torch.tensor([1000.0])
        self_.training = False
    pipe.scheduler.set_timesteps = types.MethodType(_one_step_schedule, pipe.scheduler)
    print("[1-step] scheduler patched: sigmas=[1.0], timesteps=[1000]")

control_frames = None
if args.control_video:
    import decord
    if os.path.isfile(args.ic_lora):
        pipe.load_lora(pipe.dit, lora_config=args.ic_lora)
    else:
        model_id, pattern = args.ic_lora.split(":", 1)
        pipe.load_lora(pipe.dit, lora_config=ModelConfig(model_id=model_id, origin_file_pattern=pattern))
    print(f"[ic-lora] fused {args.ic_lora}")
    vr_ctrl = decord.VideoReader(args.control_video)
    n_ctrl = len(vr_ctrl)
    control_frames = lambda start: [
        Image.fromarray(vr_ctrl[(start + i) % n_ctrl].asnumpy()).convert("RGB")
        for i in range(args.chunk_frames)
    ]
    print(f"[control] {args.control_video} ({n_ctrl} frames, looped)")

if args.time_shift_frames > 0:
    # Same hook as training (train_chunk_aware_recycle_ltx2.install_time_shift_hook_ltx2):
    # shift chunk + in_context positions; sink ref token (index-based) untouched;
    # audio idles on noise → not shifted.
    from diffsynth.pipelines.ltx2_audio_video import (
        LTX2AudioVideoUnit_NoiseInitializer,
        LTX2AudioVideoUnit_InContextVideoEmbedder,
    )
    for unit in pipe.units:
        if isinstance(unit, LTX2AudioVideoUnit_NoiseInitializer):
            _orig_noise = unit.process
            def _wrapped_noise(pipe_arg, _orig=_orig_noise, **kw):
                out = _orig(pipe_arg, **kw)
                fr = kw.get("frame_rate", 24.0) or 24.0
                out["video_positions"] = out["video_positions"].clone()
                out["video_positions"][:, 0, ...] += args.time_shift_frames / fr
                return out
            unit.process = _wrapped_noise
        if isinstance(unit, LTX2AudioVideoUnit_InContextVideoEmbedder):
            _orig_ic = unit.process
            def _wrapped_ic(pipe_arg, _orig=_orig_ic, **kw):
                out = _orig(pipe_arg, **kw)
                if out.get("in_context_video_positions") is not None:
                    fr = kw.get("frame_rate", 24.0) or 24.0
                    out["in_context_video_positions"] = out["in_context_video_positions"].clone()
                    out["in_context_video_positions"][:, 0, ...] += args.time_shift_frames / fr
                return out
            unit.process = _wrapped_ic
    print(f"[time-shift] chunk coords +{args.time_shift_frames} frames "
          f"(sink_index={args.sink_index})")

sink_img = Image.open(args.initial_ref).convert("RGB").resize((args.width, args.height)) if args.initial_ref else None
ref_for_chunk = sink_img

all_frames = []
for k in range(args.num_chunks):
    print(f"── chunk {k+1}/{args.num_chunks} ──")
    if ref_for_chunk is None:
        input_images, input_images_indexes = None, [0]
    elif args.no_sink or sink_img is None:
        input_images, input_images_indexes = [ref_for_chunk], [0]
    else:
        input_images, input_images_indexes = [ref_for_chunk, sink_img], [0, args.sink_index]

    video, audio = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_images=input_images,
        input_images_indexes=input_images_indexes,
        input_images_strength=args.input_images_strength,
        in_context_videos=[control_frames(k * (args.chunk_frames - 1))] if control_frames else None,
        in_context_downsample_factor=args.ic_downsample,
        seed=args.seed + k,
        height=args.height,
        width=args.width,
        num_frames=args.chunk_frames,
        frame_rate=args.frame_rate,
        cfg_scale=args.cfg,
        num_inference_steps=args.steps,
        tiled=True,
    )
    all_frames.extend(video if k == 0 else video[1:])
    ref_for_chunk = video[-1]
    if sink_img is None:
        sink_img = video[0]

# audio_sample_rate must be set explicitly: the writer derives it from
# audio.shape when None, which crashes for audio=None.
write_video_audio_ltx2(
    video=all_frames,
    audio=None,
    output_path=args.output,
    fps=args.frame_rate,
    audio_sample_rate=24000,
)
print(f"[done] {len(all_frames)} frames → {args.output}")
