#!/usr/bin/env bash
# Robust launcher: wait for free memory, start, and if vLLM fails at engine start (co-tenant memory changes during
# profiling), archive that attempt's log and retry. usage: launch_retry.sh NAME MEMFRAC gen_eval-args...
cd /home/xuzichuan/weightspace/experiments/loopdir_001
NAME=$1; MEM=$2; shift 2
need=$(python3 -c "print(int($MEM*40960)+3072)")
for attempt in 1 2 3 4 5; do
  while true; do
    g=""; for c in 0 1; do [ "$(nvidia-smi -i $c --query-gpu=memory.free --format=csv,noheader,nounits)" -gt "$need" ] && { g=$c; break; }; done
    [ -n "$g" ] && break; sleep 60
  done
  if docker inspect "xu_loopdir_$NAME" >/dev/null 2>&1; then
    docker logs "xu_loopdir_$NAME" > "logs/$NAME.failed_start_$(date -u +%s).log" 2>&1
    docker container rm "$(docker inspect -f {{.Id}} "xu_loopdir_$NAME")" >/dev/null
    [ -d "runs/$NAME" ] && mv "runs/$NAME" "runs/$NAME.failed_start_$(date -u +%s)"
  fi
  bash code/run_gpu.sh "$NAME" "$g" /work/code/gen_eval.py --mem "$MEM" --out "/work/runs/$NAME" "$@" >/dev/null
  sleep 240
  st=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "xu_loopdir_$NAME")
  if [ "$st" = "exited 1" ] && docker logs "xu_loopdir_$NAME" 2>&1 | grep -q "Engine core initialization failed"; then
    echo "$(date -u +%FT%TZ) $NAME attempt $attempt failed at engine start; retrying" | tee -a logs/deviations.txt
    continue
  fi
  echo "$(date -u +%FT%TZ) $NAME started on gpu $g (attempt $attempt)"; exit 0
done
echo "$NAME: gave up after 5 engine-start failures"; exit 1
