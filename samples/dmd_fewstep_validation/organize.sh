#!/bin/bash
# Organize DMD fewstep validation videos into 6 thematic folders.
# Idempotent — re-running is safe (only touches top-level *.mp4).
#
# Categories:
#   00_baseline_v2       sink_v2 teacher DMD, 4-step  (v2 + asym v2-teacher)
#   01_v2_2step          sink_v2 teacher DMD, 2-step  (old GT-noising + rollout)
#   02_teachers_50step   direct teacher inference (no student), 50-step upper bound
#   03_paper_ablation    sink_noaug teacher: asym / control / sym (4-step paper main)
#   04_sinkonly          sinkonly teacher line (student + direct teacher)
#   05_baseteacher       vanilla-base teacher line (no sink fused)
#   99_misc              didn't match any rule — eyeball later

cd "$(dirname "$0")"

mkdir -p 00_baseline_v2 01_v2_2step 02_teachers_50step \
         03_paper_ablation 04_sinkonly 05_baseteacher 99_misc

moved=0; total=0
declare -A counts

for f in *.mp4; do
    [ -f "$f" ] || continue
    total=$((total+1))

    case "$f" in
        # ─── sink_v2 teacher: 4-step variants ───
        *sink_dmd_v2_step*|*sink_dmd_asym_step*)
            target=00_baseline_v2 ;;

        # ─── sink_v2 teacher: 2-step variants (old + new rollout) ───
        *sink_dmd_2step_rollout_step*|*sink_dmd_2step_step*)
            target=01_v2_2step ;;

        # ─── teacher direct inference(没 student,50 步上界) ───
        *nostudent*50step*)
            target=02_teachers_50step ;;

        # ─── sink_noaug teacher 三档:control / sym / asym + 老 noaugteacher ───
        *dmd_noaug_control_step*|*dmd_noaug_sym_step*|*dmd_noaugteacher_asym_step*|*dmd_noaugteacher_step*)
            target=03_paper_ablation ;;

        # ─── sinkonly teacher 一族 ───
        *dmd_sinkonlyteacher_step*|*sink-sinkonly*|dmd_sinkonly_*)
            target=04_sinkonly ;;

        # ─── 原始 base teacher 一族(早期 nosink 命名 + 后期 sinkOFF) ───
        *dmd_baseteacher_step*|dmd_nosink_*|dmd_sinkOFF_*)
            target=05_baseteacher ;;

        *)
            target=99_misc ;;
    esac

    mv -n "$f" "$target/"   # -n = no overwrite
    moved=$((moved+1))
    counts[$target]=$((${counts[$target]:-0}+1))
done

echo "==== organized $moved / $total ===="
for k in 00_baseline_v2 01_v2_2step 02_teachers_50step 03_paper_ablation 04_sinkonly 05_baseteacher 99_misc; do
    n=${counts[$k]:-0}
    [ $n -gt 0 ] && echo "  $k:  $n"
done
