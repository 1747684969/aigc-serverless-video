"""MiniMax H3 云端出片端点（Serverless GPU）

流程：
  1. 容器内后台拉起无头 ComfyUI（模型来自 modal.Volume 云卷）
  2. 按官方 MiniMax H3 工作流组装 API 图并提交 /prompt
  3. 轮询 /history 直到完成，回传 H.264 MP4（视频 + 原生立体声）
  4. 任何失败都以 JSON 错误返回，本地节点可直接展示原因

部署：
    modal deploy modal/comfy_endpoint.py

鉴权（推荐开启）：
    1) 创建 secret：  modal secret create h3-endpoint-token H3_ENDPOINT_TOKEN=<你的随机串>
    2) 请求时带上头：  X-Auth-Token: <你的随机串>
"""
import os
import sys
import json
import time
import base64
import glob as globmod
import subprocess

import modal
import requests
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# 配置：按自己的环境修改
# ---------------------------------------------------------------------------
VOLUME_NAME = os.environ.get("H3_VOLUME_NAME", "h3-model-volume")   # Modal Volume 名
GPU_TYPE = os.environ.get("H3_GPU", "A100-40GB")              # A100-40GB / L40S / ...
SCALEDOWN_WINDOW = int(os.environ.get("H3_SCALEDOWN", "120"))  # 热容器保活秒数
REQUIRE_AUTH = os.environ.get("H3_REQUIRE_AUTH", "0") == "1"

app = modal.App("comfy-h3-serverless")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

COMFY_DIR = "/root/ComfyUI"
MODELS_DIR = os.path.join(COMFY_DIR, "models")
OUTPUT_DIR = os.path.join(COMFY_DIR, "output")
INPUT_DIR = os.path.join(COMFY_DIR, "input")
PORT = 8188
BASE_URL = "http://127.0.0.1:%d" % PORT
COMFY_LOG = "/root/comfyui.log"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install("fastapi", "requests")
    .run_commands(
        # 1. 克隆 ComfyUI（torch 等依赖由官方 requirements.txt 决定版本，保证与最新代码匹配）
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        "pip install -r /root/ComfyUI/requirements.txt",
        # 2. 清空官方占位文件，确保云卷能挂载到空目录
        "rm -rf /root/ComfyUI/models",
        "mkdir -p /root/ComfyUI/models",
        # 3. 视频封装插件（VHS_VideoCombine 输出 H.264 MP4）
        "cd /root/ComfyUI/custom_nodes && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    )
)


# ---------------------------------------------------------------------------
# 无头 ComfyUI 生命周期（同一容器多请求复用）
# ---------------------------------------------------------------------------
_comfy_proc = None


def _tail(path, n=3000):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[-n:]
    except Exception:
        return "(无日志)"


def _start_comfyui():
    global _comfy_proc
    for _ in range(120):  # 最多等 4 分钟
        try:
            if requests.get(BASE_URL + "/system_stats", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        if _comfy_proc is None or _comfy_proc.poll() is not None:
            if _comfy_proc is not None and _comfy_proc.poll() is not None:
                raise RuntimeError("ComfyUI 进程异常退出:\n" + _tail(COMFY_LOG))
            logf = open(COMFY_LOG, "ab")
            _comfy_proc = subprocess.Popen(
                [sys.executable, "main.py",
                 "--listen", "127.0.0.1",
                 "--port", str(PORT),
                 "--disable-auto-launch"],
                cwd=COMFY_DIR,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            print(">>> [云端] ComfyUI 启动中 ...")
        time.sleep(2)
    raise RuntimeError("ComfyUI 启动超时:\n" + _tail(COMFY_LOG))


# ---------------------------------------------------------------------------
# H3 原生画布规则：宽高必须是 32 的倍数，短边上限 768（官方 16:9 档 = 1280×736 / 0.9MP）
# ---------------------------------------------------------------------------
MIN_MEGAPIXELS = 0.9       # 低于此值自动提升到原生档位；改成 0 可关闭提升
NATIVE_SHORT_EDGE = 768


def _snap32(v):
    """对齐到 32 的倍数"""
    return max(32, int(round(v / 32.0)) * 32)


def _native_canvas(width, height):
    """对齐 32 倍数；过小的画布提升到 H3 原生档位；短边不超过 768。
    返回 (宽, 高, 说明文字或 None)。"""
    w, h = _snap32(width), _snap32(height)
    notes = []
    if (w, h) != (width, height):
        notes.append("对齐 32 倍数 %d×%d → %d×%d" % (width, height, w, h))

    mp = (w * h) / 1000000.0
    if MIN_MEGAPIXELS > 0 and mp < MIN_MEGAPIXELS:
        aspect = w / float(h)
        if 1.6 <= aspect <= 1.85:        # 约 16:9 → 官方档位
            w, h = 1280, 736
        elif 0.54 <= aspect <= 0.62:     # 约 9:16 → 官方档位
            w, h = 736, 1280
        else:                            # 其他比例按面积等比放大
            scale = (MIN_MEGAPIXELS / mp) ** 0.5
            w, h = _snap32(w * scale), _snap32(h * scale)
        notes.append("提升到原生档位 %d×%d" % (w, h))

    if min(w, h) > NATIVE_SHORT_EDGE:
        scale = NATIVE_SHORT_EDGE / float(min(w, h))
        w, h = _snap32(w * scale), _snap32(h * scale)
        notes.append("短边收敛到原生上限 %d×%d" % (w, h))

    return w, h, ("；".join(notes) if notes else None)


# ---------------------------------------------------------------------------
# 模型解析：以云卷中实际存在的文件为准，缺失时按官方文件名回退
# ---------------------------------------------------------------------------
def _find_model(subdir, preferred):
    folder = os.path.join(MODELS_DIR, subdir)
    for name in preferred:
        if name and os.path.isfile(os.path.join(folder, name)):
            return name
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".gguf")):
                return f
    return None


# ---------------------------------------------------------------------------
# 工作流执行与结果收集
# ---------------------------------------------------------------------------
def _collect_errors(entry):
    errs = []
    for nid, err in (entry.get("node_errors") or {}).items():
        errs.append("[node %s] %s" % (nid, str(err)[:1500]))
    for m in entry.get("status", {}).get("messages", []):
        if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == "execution_error":
            errs.append(str(m[1])[:1500])
    for nid, out in (entry.get("outputs") or {}).items():
        if isinstance(out, dict) and out.get("error"):
            errs.append("[node %s] %s" % (nid, str(out["error"])[:1500]))
    return errs


def _wait_history(prompt_id, timeout=840):
    deadline = time.time() + timeout
    last_status = ""
    while time.time() < deadline:
        try:
            h = requests.get(BASE_URL + "/history/" + prompt_id, timeout=10).json()
        except Exception:
            time.sleep(2)
            continue
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            errs = _collect_errors(entry)
            if errs or status.get("status_str") == "error":
                if not errs:
                    errs = ["ComfyUI 报错但未给出具体信息，请查看云端函数日志"]
                raise RuntimeError("工作流执行失败: " + " | ".join(errs))
            s = status.get("status_str", "")
            if s and s != last_status:
                print(">>> [云端] 状态:", s)
                last_status = s
        time.sleep(2)
    raise RuntimeError("工作流执行超时 (840s)")


def _run_workflow(wf):
    r = requests.post(
        BASE_URL + "/prompt",
        json={"prompt": wf, "client_id": "modal-h3"},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError("提交工作流失败: HTTP %d - %s" % (r.status_code, r.text[:800]))
    prompt_id = r.json().get("prompt_id")
    entry = _wait_history(prompt_id)

    # 从 history 输出中收集保存的文件名（兼容 SaveVideo / VHS 等不同格式）
    candidates = []
    outputs = entry.get("outputs") or {}
    for nid, out in outputs.items():
        if not isinstance(out, dict):
            continue
        for key in ("gifs", "videos", "files", "images"):
            for item in (out.get(key) or []):
                if isinstance(item, dict) and item.get("filename"):
                    p = os.path.join(OUTPUT_DIR, item.get("subfolder") or "", item["filename"])
                    if os.path.isfile(p):
                        candidates.append(p)
    # 兜底：output 目录下最新的视频文件
    if not candidates:
        candidates = [
            p for p in globmod.glob(os.path.join(OUTPUT_DIR, "**", "*"), recursive=True)
            if os.path.isfile(p) and p.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".gif"))
        ]
    if not candidates:
        raise RuntimeError("生成完成但未找到输出视频文件，history 输出: %s"
                           % json.dumps(outputs, ensure_ascii=False)[:500])
    video_path = max(candidates, key=os.path.getmtime)
    with open(video_path, "rb") as f:
        data = f.read()
    if not data:
        raise RuntimeError("输出视频为空: %s" % video_path)
    return data, video_path


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
def check_auth(request, require_auth=None, expected_token=None):
    """校验请求头 X-Auth-Token。

    独立成纯函数，便于离线单测（见 modal/test_auth.py）。
    - require_auth 为 False 时直接放行；
    - 服务端未配置 token（空值）时一律拒绝，避免"空 token 绕过"。
    """
    require_auth = REQUIRE_AUTH if require_auth is None else require_auth
    if not require_auth:
        return True
    expected = os.environ.get("H3_ENDPOINT_TOKEN", "") if expected_token is None else expected_token
    got = (getattr(request, "headers", None) or {}).get("x-auth-token", "")
    return bool(expected) and got == expected


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/ComfyUI/models": volume},
    # 容器保活：连续出片时复用已加载到显存的模型，避免重复读 26GB 云卷
    # （一次加载约 3-6 分钟纯 GPU 空转）。出一片就走人的节奏下闲置费也更省。
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=1200,
    secrets=[modal.Secret.from_name("h3-endpoint-token")] if REQUIRE_AUTH else [],
)
@modal.fastapi_endpoint(method="POST")
def generate(request: Request, payload: dict):
    if not check_auth(request):
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    try:
        return _generate(payload)
    except Exception as e:
        print("[云端错误]", repr(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


def _generate(payload):
    prompt = str(payload.get("prompt") or "A cinematic video")
    seed = int(payload.get("seed", payload.get("noise_seed", 42)))
    width = int(payload.get("width", 1280))
    height = int(payload.get("height", 736))
    width, height, canvas_note = _native_canvas(width, height)
    if canvas_note:
        print(">>> [云端] 画布调整: " + canvas_note)
    duration = float(payload.get("duration", payload.get("seconds", 5.0)))
    duration = max(1.0, min(15.0, duration))
    turbo = bool(payload.get("turbo_mode", True))

    # 官方 17k+5 帧网格（24fps）
    base = max(5, round(duration * 24))
    length = base + (5 - base % 17) % 17

    # 模型文件（以云卷实际内容为准）
    unet_name = _find_model("diffusion_models", [
        payload.get("unet_name"), "minimax_h3_fl2va_pruned_int8_convrot.safetensors"])
    clip_name = _find_model("text_encoders", [
        payload.get("clip_name"),
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"])
    vae_name = _find_model("vae", [
        payload.get("vae_name"), "minimax_h3_video_vae_fp16.safetensors"])
    audio_vae_name = _find_model("vae", [
        payload.get("audio_vae"), "minimax_h3_audio_vae_fp32.safetensors"])
    lora_name = _find_model("loras", [
        payload.get("lora_name"), "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"])

    missing = []
    if not unet_name:
        missing.append("diffusion_models/*.safetensors")
    if not clip_name:
        missing.append("text_encoders/*.safetensors")
    if not vae_name:
        missing.append("vae/minimax_h3_video_vae_fp16.safetensors")
    if not audio_vae_name:
        missing.append("vae/minimax_h3_audio_vae_fp32.safetensors")
    if missing:
        return JSONResponse(status_code=400, content={
            "error": "云卷 %s 缺少模型: %s。请先运行 modal/download_models.py 下载模型。"
                     % (VOLUME_NAME, ", ".join(missing))})

    use_lora = turbo and lora_name is not None
    steps = int(payload.get("turbo_steps", 8)) if use_lora else int(payload.get("steps", 20))
    if turbo and not use_lora:
        print(">>> [云端] 警告: turbo 开启但云卷中没有 LoRA，退回标准 20 步。")

    print(">>> [云端] 生成请求 | %dx%d | %d 帧 | %d 步 | seed=%d"
          % (width, height, length, steps, seed))
    _start_comfyui()

    # 首尾帧（本地节点的 PNG base64）写入 input 目录
    ff_name = lf_name = None
    if payload.get("first_frame_b64"):
        ff_name = "modal_ff_%d.png" % seed
        with open(os.path.join(INPUT_DIR, ff_name), "wb") as f:
            f.write(base64.b64decode(payload["first_frame_b64"]))
    if payload.get("last_frame_b64"):
        lf_name = "modal_lf_%d.png" % seed
        with open(os.path.join(INPUT_DIR, lf_name), "wb") as f:
            f.write(base64.b64decode(payload["last_frame_b64"]))

    # 组装官方 MiniMax H3 工作流（API 格式）
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        "vae_v": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "vae_a": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }

    model_ref = ["unet", 0]
    if use_lora:
        wf["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model_ref,
            "lora_name": lora_name,
            "strength_model": float(payload.get("turbo_model_strength", 1.0)),
        }}
        model_ref = ["lora", 0]

    cond_inputs = {
        "clip": ["clip", 0],
        "vae": ["vae_v", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
    }
    if ff_name:
        wf["ff"] = {"class_type": "LoadImage", "inputs": {"image": ff_name}}
        cond_inputs["first_frame"] = ["ff", 0]
    if lf_name:
        wf["lf"] = {"class_type": "LoadImage", "inputs": {"image": lf_name}}
        cond_inputs["last_frame"] = ["lf", 0]

    wf["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}
    wf["sched"] = {"class_type": "BasicScheduler", "inputs": {
        "model": model_ref, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    wf["guider"] = {"class_type": "BasicGuider", "inputs": {
        "model": model_ref, "conditioning": ["cond", 0]}}
    wf["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
        "sigmas": ["sched", 0], "latent_image": ["cond", 1]}}
    wf["decode_v"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["sample", 0], "vae": ["vae_v", 0]}}
    wf["decode_a"] = {"class_type": "VAEDecodeAudio", "inputs": {
        "samples": ["sample", 0], "vae": ["vae_a", 0]}}

    # 动态选择出片节点：优先核心 CreateVideo + SaveVideo，回退 VHS_VideoCombine
    try:
        obj_info = requests.get(BASE_URL + "/object_info", timeout=30).json()
    except Exception as e:
        raise RuntimeError("获取 /object_info 失败: %s" % e)
    if "SaveVideo" in obj_info and "CreateVideo" in obj_info:
        wf["create"] = {"class_type": "CreateVideo", "inputs": {
            "images": ["decode_v", 0], "audio": ["decode_a", 0], "fps": 24, "bit_depth": 8}}
        wf["save"] = {"class_type": "SaveVideo", "inputs": {
            "video": ["create", 0],
            "filename_prefix": "minimax_h3_%d" % seed,
            "format": "auto",
            "codec": "auto"}}
        print(">>> [云端] 使用核心节点 CreateVideo/SaveVideo 出片")
    elif "VHS_VideoCombine" in obj_info:
        wf["save"] = {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["decode_v", 0],
            "audio": ["decode_a", 0],
            "frame_rate": 24.0,
            "loop_count": 0,
            "filename_prefix": "minimax_h3_%d" % seed,
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True}}
        print(">>> [云端] 使用 VHS_VideoCombine 出片")
    else:
        raise RuntimeError("云端 ComfyUI 缺少视频封装节点（SaveVideo/CreateVideo 与 VHS_VideoCombine 均不可用）")

    data, video_path = _run_workflow(wf)
    print(">>> [云端] 出片完成: %s (%d 字节)" % (video_path, len(data)))
    return Response(content=data, media_type="video/mp4")
