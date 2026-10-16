#!/usr/bin/env python3
"""
Antigravity Stop Hook Handler for Ralph Loop Engine
Contract:
- Reads JSON payload from stdin (terminationReason, transcriptPath, conversationId, etc.)
- Checks .agents/loop/state.json
- If loop is active and completion promise is not met, returns {"decision": "continue", "reason": "..."}
- Otherwise returns {"decision": "stop"}
"""

import sys
import json
import os
from pathlib import Path

def main():
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input.strip() else {}
    except Exception:
        payload = {}

    workspace_paths = payload.get("workspacePaths", ["."])
    workspace_dir = Path(workspace_paths[0]) if workspace_paths else Path(".")
    loop_dir = workspace_dir / ".agents" / "loop"
    state_file = loop_dir / "state.json"

    # Default exit decision
    stop_decision = {"decision": "stop"}

    if not state_file.exists():
        print(json.dumps(stop_decision))
        return

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        print(json.dumps(stop_decision))
        return

    if not state.get("enabled", False):
        print(json.dumps(stop_decision))
        return

    current_iter = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations", 10)
    promise = state.get("completion_promise", "COMPLETED")
    transcript_path = payload.get("transcriptPath", "")

    # Check transcript for promise token if available
    promise_found = False
    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as tf:
                # Read last 50 lines to detect completion promise
                lines = tf.readlines()[-50:]
                for line in lines:
                    if promise in line:
                        promise_found = True
                        break
        except Exception:
            pass

    # Check circuit breaker conditions
    if promise_found:
        state["enabled"] = False
        state["status"] = "success"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(json.dumps({"decision": "stop", "reason": f"Completion promise '{promise}' fulfilled."}))
        return

    if current_iter >= max_iter:
        state["enabled"] = False
        state["status"] = "max_iterations_reached"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(json.dumps({"decision": "stop", "reason": f"Max iterations ({max_iter}) reached."}))
        return

    # Increment iteration and continue loop
    current_iter += 1
    state["current_iteration"] = current_iter
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    task_desc = state.get("task_description", "Continue working on the goal")
    continue_resp = {
        "decision": "continue",
        "reason": f"[Ralph Loop #{current_iter}/{max_iter}] 목표가 아직 완수되지 않았습니다. 완료 약속({promise})이 충족될 때까지 다음 작업을 이어서 수행하세요. 작업 목표: {task_desc}"
    }
    print(json.dumps(continue_resp, ensure_ascii=False))

if __name__ == "__main__":
    main()
