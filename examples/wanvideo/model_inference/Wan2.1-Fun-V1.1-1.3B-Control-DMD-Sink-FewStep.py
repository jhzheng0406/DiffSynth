"""
Few-step chain inference for the DMD-distilled sink student LoRA.

Model stack (must mirror DMD training in train_dmd.py):
    base  +  sink LoRA (fused)  +  student LoRA (fused)
The student LoRA was trained on top of the sink-fused base, so BOTH must be
loaded for inference — student LoRA alone is meaningless.

Key few-step settings (differ from the 50-step sink inference):
    num_inference_steps = 4        # the whole point of DMD
    sigma_shift         = 5.0      # must match training (scheduler shift)
    cfg_scale           = 1.0      # DMD student is distilled WITHOUT CFG;
                                   # cfg_scale>1 applies guidance it never saw

A/B usage during DMD training:
    # 4-step student at checkpoint step-200
    python ...DMD-Sink-FewStep.py --student step-200 --steps 4
    # sanity: same student at 50 steps (should look better; isolates whether
    #         few-step is the limiter vs the student being broken)
    python ...DMD-Sink-FewStep.py --student step-200 --steps 50
    # baseline: sink only, no student (set --student "")
    python ...DMD-Sink-FewStep.py --student "" --steps 4
"""
import argparse, os, torch
from PIL import Image
from diffsynth.utils.data import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--sink_lora",   default="models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors",
                    help="Frozen sink LoRA — same one used as DMD teacher.")
parser.add_argument("--student", default="step-200",
                    help="DMD student checkpoint name (without dir/ext), or a "
                         "full path, or empty string for sink-only baseline.")
parser.add_argument("--student_dir", default="models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd",
                    help="Directory holding step-*.safetensors student checkpoints.")
parser.add_argument("--steps",       type=int, default=4)
parser.add_argument("--first_chunk_steps", type=int, default=None,
                    help="Override --steps for chunk 0 only (ASD trick: use more "
                         "steps on the first chunk to bootstrap a clean reference; "
                         "later chunks use --steps as usual). Default: same as --steps.")
parser.add_argument("--sigma_shift", type=float, default=5.0)
parser.add_argument("--cfg_scale",   type=float, default=None,
                    help="Default auto: 1.0 with a DMD student (distilled to be "
                         "CFG-free), 5.0 for the sink-only baseline (undistilled "
                         "model needs CFG even at few steps). Pass to override.")
parser.add_argument("--no_recent",   action="store_true",
                    help="Sink-only conditioning: don't pass the recent frame. "
                         "Use for sinkonly models (trained without recent).")
parser.add_argument("--num_chunks",  type=int, default=10)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--control_video", default="asset/pose_loop.mp4")
parser.add_argument("--initial_ref",   default="asset/6.png")
parser.add_argument("--save_dir",       default="samples/dmd_fewstep_validation")
parser.add_argument("--prompt",         default=None,
                    help="Override the hardcoded PROMPT below.")
parser.add_argument("--tag",            default="",
                    help="Optional short string appended to the output filename "
                         "(e.g. 'beach' when you switch ref+prompt) so different "
                         "scenes don't collide.")
args = parser.parse_args()

HEIGHT, WIDTH, CHUNK_FRAMES, FPS = 832, 480, 49, 16
PROMPT = args.prompt if args.prompt else (
    "动漫风格，紫色短发少女正在轻盈起舞。她头戴黑色发箍，身穿白色连衣裙，外搭黑色背心，"
    "胸前系着粉色蝴蝶结。背景是粉色的大圆圈，画面简洁柔和。"
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# ---------------------------------------------------------------------------
# Resolve student checkpoint path
# ---------------------------------------------------------------------------
student_path = ""
if args.student.strip():
    s = args.student.strip()
    if os.path.isfile(s):
        student_path = s
    else:
        # Accept shorthand: "1200", "1200_ema", "step-1200", "step-1200.safetensors", ...
        name = s
        if not name.startswith("step-") and not name.endswith(".safetensors") \
                and os.sep not in name and name and name[0].isdigit():
            name = f"step-{name}"
        if not name.endswith(".safetensors"):
            name = f"{name}.safetensors"
        student_path = os.path.join(args.student_dir, name)
    if not os.path.isfile(student_path):
        raise FileNotFoundError(f"Student LoRA not found: {student_path}")

# Resolve cfg_scale default by mode: distilled student is CFG-free (1.0); the
# undistilled sink-only baseline needs normal CFG (5.0) for a fair few-step run.
if args.cfg_scale is None:
    args.cfg_scale = 1.0 if student_path else 5.0

# ---------------------------------------------------------------------------
# Build pipe and stack LoRAs: base → sink (fused) → student (fused)
# ---------------------------------------------------------------------------
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Skip sink fusion for students trained on the VANILLA base (the base-teacher
# experiment): pass --sink_lora none. Otherwise the student LoRA would sit on
# the wrong base.
if args.sink_lora and args.sink_lora.lower() != "none":
    print(f"[lora] fusing sink:    {args.sink_lora}")
    pipe.load_lora(pipe.dit, lora_config=args.sink_lora, alpha=1.0)
else:
    print("[lora] NO sink fused (vanilla base — for base-teacher student)")
if student_path:
    print(f"[lora] fusing student: {student_path}")
    pipe.load_lora(pipe.dit, lora_config=student_path, alpha=1.0)
else:
    print("[lora] no student LoRA (baseline)")

print(f"[infer] steps={args.steps}  sigma_shift={args.sigma_shift}  cfg_scale={args.cfg_scale}")

# ---------------------------------------------------------------------------
# Control video: slice per chunk with 1-frame overlap (matches training)
# ---------------------------------------------------------------------------
total_frames = CHUNK_FRAMES + (args.num_chunks - 1) * (CHUNK_FRAMES - 1)
full_ctrl = VideoData(args.control_video, height=HEIGHT, width=WIDTH)
all_ctrl_frames = [full_ctrl[i] for i in range(min(len(full_ctrl), total_frames))]
while len(all_ctrl_frames) < total_frames:
    all_ctrl_frames += all_ctrl_frames
all_ctrl_frames = all_ctrl_frames[:total_frames]

sink_img = Image.open(args.initial_ref).convert("RGB")

# ---------------------------------------------------------------------------
# Chain loop — sink fixed; recent = previous chunk's last frame
# ---------------------------------------------------------------------------
all_frames = []
ref_for_chunk = sink_img
for k in range(args.num_chunks):
    print(f"── chunk {k+1}/{args.num_chunks} ──")
    frame_start = k * (CHUNK_FRAMES - 1)
    ctrl_chunk  = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

    chunk_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        control_video=ctrl_chunk,
        reference_image=(None if args.no_recent else ref_for_chunk),  # recent (skipped for sinkonly)
        sink_reference_image=sink_img,         # sink (constant)
        height=HEIGHT, width=WIDTH,
        num_frames=CHUNK_FRAMES,
        num_inference_steps=(args.first_chunk_steps if (k == 0 and args.first_chunk_steps) else args.steps),
        sigma_shift=args.sigma_shift,
        cfg_scale=args.cfg_scale,
        seed=args.seed + k,
        tiled=True,
    )
    all_frames.extend(chunk_frames if k == 0 else chunk_frames[1:])
    ref_for_chunk = chunk_frames[-1]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(args.save_dir, exist_ok=True)

# Filename encodes the FULL model stack so different experiments never collide:
#   sink<on|off>   — was the sink LoRA fused into base
#   exp<dirname>   — which training run the student came from (v2 / asym /
#                    baseteacher / 2step ...), so teacher recipe is identifiable
#   <step|none>    — which student checkpoint (none = no student / baseline)
sink_on = bool(args.sink_lora) and args.sink_lora.lower() != "none"
if sink_on:
    # Identify WHICH sink LoRA (sink_v2 / sink_noaug / sinkonly ...) so direct
    # teacher inferences (no student) don't collide.
    sink_name = os.path.basename(os.path.dirname(os.path.normpath(args.sink_lora)))
    sink_name = sink_name.replace("Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_", "")
    sink_tag = f"sink-{sink_name}"
else:
    sink_tag = "sinkOFF"
recent_tag = "_norecent" if args.no_recent else ""
if student_path:
    exp_tag = f"exp-{os.path.basename(os.path.normpath(args.student_dir))}"
    step_tag = os.path.splitext(os.path.basename(student_path))[0]   # e.g. step-1200
    student_tag = f"{exp_tag}_{step_tag}"
else:
    student_tag = "nostudent"   # sink-only or pure-base baseline

scene_tag = f"_{args.tag}" if args.tag else ""
fname = (f"dmd_{sink_tag}{recent_tag}_{student_tag}_{args.steps}step_cfg{args.cfg_scale}"
         f"_chunks{args.num_chunks}x{CHUNK_FRAMES}_{HEIGHT}x{WIDTH}_seed{args.seed}{scene_tag}.mp4")
out_path = os.path.join(args.save_dir, fname)
save_video(all_frames, out_path, fps=FPS, quality=5)
print(f"\nSaved {len(all_frames)} frames → {out_path}")
