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
parser.add_argument("--base_model_id", default="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                    help="Base Fun-Control model_id. Set 'PAI/Wan2.1-Fun-V1.1-14B-Control' "
                         "to load a 14B sink LoRA / student.")
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
parser.add_argument("--no_sink",     action="store_true",
                    help="Recent-only (i2v chaining) conditioning: don't pass the "
                         "sink reference. Use for with_ref models (trained without "
                         "sink). --initial_ref still seeds chunk-0's recent frame.")
parser.add_argument("--num_chunks",  type=int, default=10)
parser.add_argument("--plp", action="store_true",
                    help="Persistent Latent Propagation (EverAnimate-style): feed "
                         "the previous chunk's last latent frame directly as the "
                         "recent reference, skipping the lossy decode→re-encode. "
                         "Requires a PLP-trained checkpoint for train/test match.")
parser.add_argument("--vel_recent", action="store_true",
                    help="PLP-v2 velocity-aware recent: carry the prev chunk's LAST TWO "
                         "latent frames as the recent ref (motion context). MUST match "
                         "how the student was trained (--vel_recent).")
parser.add_argument("--causal", action="store_true",
                    help="One-Forcing causal-AR baseline: run the student with the "
                         "block-causal self-attn mask it was trained with (each "
                         "latent frame attends only to the ref prefix + earlier "
                         "frames). MUST match how the student was trained.")
parser.add_argument("--causal_block_frames", type=int, default=1,
                    help="Block-causal granularity (latent frames); match training.")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--height",      type=int, default=832, help="Frame height (832 portrait default; 480 for landscape).")
parser.add_argument("--width",       type=int, default=480, help="Frame width (480 portrait default; 832 for landscape).")
parser.add_argument("--control_video", default="asset/pose_loop.mp4")
parser.add_argument("--initial_ref",   default="asset/6.png")
parser.add_argument("--t2v",           action="store_true",
                    help="Text-to-video bootstrap: chunk 0 gets NO reference "
                         "image at all (pure prompt + control), then its first "
                         "generated frame becomes the sink for later chunks. "
                         "--initial_ref is ignored.")
parser.add_argument("--save_dir",       default="samples/dmd_fewstep_validation")
parser.add_argument("--prompt",         default=None,
                    help="Override the hardcoded PROMPT below.")
parser.add_argument("--tag",            default="",
                    help="Optional short string appended to the output filename "
                         "(e.g. 'beach' when you switch ref+prompt) so different "
                         "scenes don't collide.")
args = parser.parse_args()

HEIGHT, WIDTH, CHUNK_FRAMES, FPS = args.height, args.width, 49, 16
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
        ModelConfig(model_id=args.base_model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id=args.base_model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=args.base_model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id=args.base_model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
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

if args.causal:
    # Causal-AR baseline: model_fn_wan_video builds+attaches the block-causal mask
    # per forward (shape-cached) once this attr is set on the student's DiT.
    pipe.dit._causal_block_frames = args.causal_block_frames
    print(f"[causal] block-causal inference ON (block_frames={args.causal_block_frames})")

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

sink_img = None if args.t2v else Image.open(args.initial_ref).convert("RGB")

# ---------------------------------------------------------------------------
# Chain loop — sink fixed; recent = previous chunk's last frame
# (t2v: chunk 0 has no refs; its first frame is promoted to sink afterwards)
# ---------------------------------------------------------------------------
import time as _time
# --- isolate VAE time: wrap vae.{encode,decode,*_framewise} to accumulate.
# dit(no-vae) = total - vae. (T5 text-encode left IN; wrapping it caused issues.)
_vae_acc = [0.0]
def _wrap(_fn, _acc):
    def _w(*a, **k):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _ts = _time.time()
        r = _fn(*a, **k)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _acc[0] += _time.time() - _ts
        return r
    return _w
for _m in ("encode", "decode", "encode_framewise", "decode_framewise"):
    if hasattr(pipe.vae, _m):
        setattr(pipe.vae, _m, _wrap(getattr(pipe.vae, _m), _vae_acc))

chunk_times = []      # total per-chunk
dit_times   = []      # per-chunk minus VAE enc/dec
all_frames = []
ref_for_chunk = sink_img
recent_latent_next = None      # PLP: prev chunk's last latent frame (None on chunk 0)
for k in range(args.num_chunks):
    _vae_acc[0] = 0.0
    print(f"── chunk {k+1}/{args.num_chunks} ──")
    frame_start = k * (CHUNK_FRAMES - 1)
    ctrl_chunk  = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

    if torch.cuda.is_available(): torch.cuda.synchronize()
    _t0 = _time.time()
    chunk_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        control_video=ctrl_chunk,
        reference_image=(None if args.no_recent else ref_for_chunk),  # recent pixel (CLIP; latent path uses recent_latent under --plp)
        sink_reference_image=(None if args.no_sink else sink_img),     # sink (skipped for recent-only/with_ref)
        recent_latent=(None if args.no_recent else (recent_latent_next if args.plp else None)),  # PLP: prev chunk's latent; suppressed under --no_recent (sink-only) and on chunk 0
        height=HEIGHT, width=WIDTH,
        num_frames=CHUNK_FRAMES,
        num_inference_steps=(args.first_chunk_steps if (k == 0 and args.first_chunk_steps) else args.steps),
        sigma_shift=args.sigma_shift,
        cfg_scale=args.cfg_scale,
        seed=args.seed + k,
        tiled=True,
        return_latents=(args.plp and not args.no_recent),   # no point keeping latents when recent is suppressed
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _dt = _time.time() - _t0
    if args.plp and not args.no_recent:
        chunk_frames, _zk = chunk_frames           # unpack (video, latents)
        _nr = 2 if args.vel_recent else 1          # PLP-v2: carry last TWO frames (velocity)
        recent_latent_next = _zk[:, :, -_nr:]       # last latent frame(s) → next chunk's recent
    _vae_dt = _vae_acc[0]
    _dit_dt = _dt - _vae_dt
    chunk_times.append(_dt)
    dit_times.append(_dit_dt)
    nfe = args.first_chunk_steps if (k == 0 and args.first_chunk_steps) else args.steps
    print(f"   chunk {k+1}: total={_dt*1000:.1f}ms  vae={_vae_dt*1000:.1f}ms  "
          f"dit(no-vae)={_dit_dt*1000:.1f}ms  (NFE={nfe})")

    all_frames.extend(chunk_frames if k == 0 else chunk_frames[1:])
    ref_for_chunk = chunk_frames[-1]
    if args.t2v and k == 0:
        sink_img = chunk_frames[0]

# ── timing summary: steady-state = chunk 1+ (excludes chunk-0 warmup) ──
# DiT(no-vae) is the part that scales with NFE; total includes VAE enc/dec.
import statistics as _st
def _ms(x): return f"{x*1000:.1f}ms"
_tot_s = chunk_times[1:] if len(chunk_times) > 1 else chunk_times
_dit_s = dit_times[1:]   if len(dit_times)   > 1 else dit_times
_tot_m, _dit_m = _st.mean(_tot_s), _st.mean(_dit_s)
_dit_std = _st.pstdev(_dit_s) if len(_dit_s) > 1 else 0.0
_line = (f"[timing] steps={args.steps}  chunks={args.num_chunks}  "
         f"per-chunk: total_mean={_ms(_tot_m)}  "
         f"DiT_no_vae_mean={_ms(_dit_m)} (std {_ms(_dit_std)}; incl text)  "
         f"vae_mean={_ms(_tot_m - _dit_m)}  chunk0_warmup={_ms(chunk_times[0])}  (n={len(_dit_s)})")
print("\n" + _line)
# Append a record line to the timing log (auto-recorded per run).
_tlog = "notes/analysis/inference_timing.txt"
try:
    _stu = os.path.basename(os.path.normpath(args.student_dir)) if student_path else "teacher(no student)"
    with open(_tlog, "a") as _f:
        _f.write(f"\n{args.steps:>3}-step | {_stu} | DiT(no-vae) {_ms(_dit_m)} "
                 f"| total {_ms(_tot_m)} | vae {_ms(_tot_m - _dit_m)} "
                 f"| {HEIGHT}x{WIDTH} chunks={args.num_chunks} | {_line}")
    print(f"[timing] appended → {_tlog}")
except Exception as _e:
    print(f"[timing] could not write log: {_e}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(args.save_dir, exist_ok=True)

# Filename encodes the FULL model stack so different experiments never collide:
#   sink<on|off>   — was the sink LoRA fused into base
#   exp<dirname>   — which training run the student came from (v2 / asym /
#                    baseteacher / 2step ...), so teacher recipe is identifiable
#   <step|none>    — which student checkpoint (none = no student / baseline)
import re as _re
def _short(name):   # strip verbose, redundant base-model prefixes from dir names
    name = _re.sub(r"Wan2\.1-Fun-V1\.1-(?:1\.3B|14B)-Control_lora_", "", name)
    name = name.replace("cartoon_", "")
    name = _re.sub(r"^wan(?:1\.3b|14b)_", "", name)   # student dir prefix
    return name
sink_on = bool(args.sink_lora) and args.sink_lora.lower() != "none"
if sink_on:
    # Identify WHICH sink LoRA (sink_v2 / sink_recycle_v1 / sinkonly ...) so direct
    # teacher inferences (no student) don't collide.
    sink_tag = "sink-" + _short(os.path.basename(os.path.dirname(os.path.normpath(args.sink_lora))))
else:
    sink_tag = "sinkOFF"
recent_tag = "_norecent" if args.no_recent else ""
nosink_tag = "_nosink" if args.no_sink else ""    # sink LoRA loaded but sink ref withheld
plp_tag    = "_plp" if args.plp else ""           # persistent latent propagation on
if student_path:
    exp_tag = f"exp-{_short(os.path.basename(os.path.normpath(args.student_dir)))}"
    step_tag = os.path.splitext(os.path.basename(student_path))[0]   # e.g. step-1200
    student_tag = f"{exp_tag}_{step_tag}"
else:
    student_tag = "nostudent"   # sink-only or pure-base baseline

mode_tag = "_t2v" if args.t2v else ""
if args.tag:
    # Caller supplied a short label → just use it (keeps filenames human-short).
    fname = _re.sub(r"[^A-Za-z0-9_.-]+", "-", args.tag) + ".mp4"
else:
    # No tag → full descriptive name (encodes the model stack so runs don't collide).
    fname = (f"dmd_{sink_tag}{recent_tag}{nosink_tag}{plp_tag}{mode_tag}_{student_tag}_{args.steps}step_cfg{args.cfg_scale}"
             f"_chunks{args.num_chunks}x{CHUNK_FRAMES}_{HEIGHT}x{WIDTH}_seed{args.seed}.mp4")
# Guard against >255-byte filenames (NAME_MAX) — ffmpeg fails with "File name too long".
if len(fname) > 200:
    fname = fname[:196] + ".mp4"
out_path = os.path.join(args.save_dir, fname)
# Generation is expensive — don't lose it if the ffmpeg encode dies (broken
# pipe, e.g. ffmpeg OOM-killed under node memory pressure). Try mp4 with
# explicit libx264/yuv420p; on ANY failure, dump PNG frames so the result is kept.
import tempfile, shutil as _shutil
try:
    # ffmpeg muxing mp4 straight onto NFS (/mnt/vita) can break the write pipe
    # (seek on the moov atom). Encode to LOCAL /tmp first, then move to the NAS.
    _tmp_mp4 = os.path.join(tempfile.gettempdir(), os.path.basename(out_path))
    save_video(all_frames, _tmp_mp4, fps=FPS, quality=5,
               ffmpeg_params=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    _shutil.move(_tmp_mp4, out_path)
    print(f"\nSaved {len(all_frames)} frames → {out_path}")
except Exception as _e:
    frames_dir = out_path[:-4] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for _i, _f in enumerate(all_frames):
        _f.save(os.path.join(frames_dir, f"{_i:05d}.png"))
    print(f"\n[warn] mp4 encode failed ({_e!r}); saved {len(all_frames)} PNG frames → {frames_dir}\n"
          f"       make an mp4 later: ffmpeg -framerate {FPS} -i {frames_dir}/%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p {out_path}")
