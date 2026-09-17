# -*- coding: utf-8 -*-
"""Image to Video (MiniMax H3) — Modal 云端出片节点

优化要点：
  1. 结果复用：参数与上次完全一致时不重复调用云端（省一次 GPU 费用）
  2. 客户端超时 1500 秒 > 云端 1200 秒，避免丢弃已付费的生成结果
  3. 系统代理故障时自动改直连重试（v2rayN/Clash 抽风不再中断出片）
  4. video_path 输出完整路径，可直接喂给 Video2X
  5. PyAV 解码改为预分配数组，峰值内存降低约 30-50%
  6. 自动探测成片分辨率/帧率/帧数/音频流，出片即可核对
"""
import os
import io
import sys
import json
import time
import base64
import shutil
import hashlib
import subprocess
import tempfile
import requests
import folder_paths
import torch
import numpy as np
from PIL import Image

# --- 可调参数 ---------------------------------------------------------------
# Modal 端点地址：请改成你自己 modal deploy 输出的地址
# 也可用环境变量注入：set MODAL_H3_URL=https://<workspace>--<app>-<fn>.modal.run
MODAL_URL_DEFAULT = os.environ.get(
    "MODAL_H3_URL",
    "https://<your-workspace>--comfy-h3-serverless-generate.modal.run",
)

# 端点鉴权 token（服务端开启 H3_REQUIRE_AUTH=1 时必填）
# 从环境变量读取，避免把凭证写进代码或提交到仓库
MODAL_AUTH_TOKEN = os.environ.get("MODAL_H3_TOKEN", "")

CLOUD_TIMEOUT = 1500     # 单次请求最长等待秒数（必须大于云端 timeout=1200）
REUSE_RESULT = True      # 参数未变时复用本地成片，不重复调用云端


def _auth_headers():
    """构造鉴权请求头；未配置 token 时返回空 dict（兼容服务端未开启鉴权）"""
    return {"X-Auth-Token": MODAL_AUTH_TOKEN} if MODAL_AUTH_TOKEN else {}


def tensor_to_base64(image_tensor):
    """将 ComfyUI 的图像 Tensor 转为 base64 字符串发送给云端"""
    if image_tensor is None:
        return None
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    np_img = (image_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===========================================================================
# 网络层
# ===========================================================================
def _post_to_cloud(url, payload, timeout):
    """提交请求；系统代理不可用时自动改为直连重试"""
    headers = _auth_headers()
    try:
        return requests.post(url, json=payload, timeout=timeout, headers=headers)
    except requests.exceptions.ProxyError as e:
        print(f">>> 检测到系统代理异常 ({e})，改为直连重试...")
        session = requests.Session()
        session.trust_env = False          # 忽略 HTTP_PROXY / 系统代理设置
        return session.post(url, json=payload, timeout=timeout, headers=headers,
                            proxies={"http": None, "https": None})


def _payload_fingerprint(payload):
    """只对影响生成结果的字段做指纹，方便判断能否复用已有成片"""
    keys = ("prompt", "seed", "steps", "width", "height", "duration",
            "turbo_mode", "lora_name", "turbo_model_strength",
            "unet_name", "clip_name", "vae_name", "audio_vae",
            "first_frame_b64", "last_frame_b64")
    m = hashlib.sha256()
    for k in keys:
        m.update(str(payload.get(k)).encode("utf-8", "ignore"))
        m.update(b"\x00")
    return m.hexdigest()[:32]


def _looks_like_video(content):
    """通过魔数粗略判断是否为视频容器（MP4/MOV/WebM/MKV/AVI/FLV）"""
    content = content.lstrip()
    if len(content) < 16:
        return False
    head = content[:16]
    if head[4:8] in (b"ftyp", b"styp"):
        return True
    if head[:4] in (b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"moof", b"styp"):
        return True
    if head[:4] == b"\x1aE\xdf\xa3":
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True
    if head[:3] == b"FLV":
        return True
    return False


def _extract_video_bytes(content):
    """解析云端响应：支持原始视频字节，以及 JSON 包裹的 base64 / 视频 URL"""
    if not content:
        raise RuntimeError("云端返回空内容")
    stripped = content.lstrip()
    if stripped[:1] == b"{":
        try:
            data = json.loads(content.decode("utf-8"))
        except Exception:
            return content
        if isinstance(data, dict):
            for key in ("video", "video_b64", "video_base64", "base64", "data", "result"):
                val = data.get(key)
                if isinstance(val, str) and val:
                    try:
                        raw = base64.b64decode(val)
                        if _looks_like_video(raw):
                            return raw
                    except Exception:
                        pass
            for key in ("url", "video_url", "download_url"):
                val = data.get(key)
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    r = requests.get(val, timeout=600)
                    r.raise_for_status()
                    if r.content:
                        return r.content
    return content


# ===========================================================================
# 视频探测与解码
# ===========================================================================
def _probe_video(video_path):
    """用 PyAV 探测成片参数，返回 dict 或 None"""
    try:
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            return {
                "width": stream.width,
                "height": stream.height,
                "fps": float(stream.average_rate) if stream.average_rate else 24.0,
                "frames": int(stream.frames) if stream.frames else 0,
                "audio": len(container.streams.audio),
                "codec": stream.codec_context.name,
            }
    except Exception:
        return None


def _decode_with_pyav(video_path):
    """PyAV 解码：预分配 float32 数组逐帧写入，避免“列表 + 整体转换”的双份内存峰值"""
    import av
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        buf = None
        n = 0
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            if buf is None:
                cap = int(stream.frames) if stream.frames else 0
                buf = np.empty((cap if cap > 0 else 256,) + arr.shape, dtype=np.float32)
            elif n >= buf.shape[0]:
                buf = np.concatenate([buf, np.empty_like(buf)], axis=0)   # 容量不足则翻倍
            buf[n] = arr
            buf[n] /= 255.0
            n += 1
        if n == 0:
            raise RuntimeError("未解码到任何帧")
        return torch.from_numpy(buf[:n])
    finally:
        container.close()


def _find_ffmpeg():
    """在 ComfyUI 便携包/系统 PATH 中查找 ffmpeg 可执行文件"""
    candidates = ["ffmpeg", "ffmpeg.exe"]
    comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    extra_paths = [
        os.path.join(comfy_root, "ffmpeg", "bin"),
        os.path.join(comfy_root, "ffmpeg"),
        os.path.join(comfy_root, "ComfyUI", "ffmpeg", "bin"),
        os.path.join(comfy_root, "python_embeded", "ffmpeg", "bin"),
        os.path.join(os.path.dirname(sys.executable), "ffmpeg", "bin"),
        os.path.join(os.path.dirname(sys.executable)),
    ]
    search_paths = extra_paths + os.environ.get("PATH", "").split(os.pathsep)
    for folder in search_paths:
        if not folder:
            continue
        for name in candidates:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    return "ffmpeg"  # 回退到 PATH 命令


def _decode_with_ffmpeg(video_path):
    """兜底：用 ffmpeg 导出 PNG 序列再读入（本机未装 ffmpeg 时会被跳过）"""
    ffmpeg = _find_ffmpeg()
    tmpdir = tempfile.mkdtemp(prefix="modal_h3_frames_")
    try:
        pattern = os.path.join(tmpdir, "frame_%06d.png")
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-pix_fmt", "rgb24",
            "-start_number", "0",
            pattern,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "ignore").strip()[-2000:])
        frames = []
        for fname in sorted(os.listdir(tmpdir)):
            if fname.lower().endswith(".png"):
                img = Image.open(os.path.join(tmpdir, fname)).convert("RGB")
                frames.append(np.array(img, dtype=np.float32) / 255.0)
        if not frames:
            raise RuntimeError("ffmpeg 未提取到任何帧")
        return torch.from_numpy(np.stack(frames, axis=0))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def decode_video_frames(video_path):
    """跨平台通用视频解码器：PyAV → OpenCV → ffmpeg，全部失败才报错并给出详情"""
    if not os.path.isfile(video_path):
        raise RuntimeError(f"视频文件不存在: {video_path}")
    if os.path.getsize(video_path) == 0:
        raise RuntimeError(f"视频文件为空: {video_path}")

    errors = []

    # 方案 1：PyAV（内存优化版，ComfyUI 自带）
    try:
        return _decode_with_pyav(video_path)
    except Exception as e:
        errors.append(f"PyAV: {e}")

    # 方案 2：OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            return torch.from_numpy(np.array(frames, dtype=np.float32) / 255.0)
        errors.append("OpenCV: 未读取到任何帧")
    except Exception as e:
        errors.append(f"OpenCV: {e}")

    # 方案 3：ffmpeg
    try:
        return _decode_with_ffmpeg(video_path)
    except Exception as e:
        errors.append(f"ffmpeg: {e}")

    err_detail = " | ".join(errors)
    raise RuntimeError(
        f"未能解析视频文件: {video_path}。请确保本地支持基本视频解码。"
        f"已尝试 PyAV/OpenCV/ffmpeg 均失败。详情: {err_detail}"
    )


# ===========================================================================
# H3 原生画布规则：宽高必须是 32 的倍数，短边上限 768（官方 16:9 档 = 1280×736）
# ===========================================================================
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
        notes.append(f"对齐 32 倍数 {width}×{height} → {w}×{h}")

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
        notes.append(f"提升到原生档位 {w}×{h}")

    if min(w, h) > NATIVE_SHORT_EDGE:
        scale = NATIVE_SHORT_EDGE / float(min(w, h))
        w, h = _snap32(w * scale), _snap32(h * scale)
        notes.append(f"短边收敛到原生上限 {w}×{h}")

    return w, h, ("；".join(notes) if notes else None)


# ===========================================================================
# 节点
# ===========================================================================
class ModalMiniMaxH3Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Realistic live-action cinematic look, action movie trailer: practical film photography style, a post-rain dusk metropolis, anamorphic lens, shallow depth of field, film grain, city volumetric fog, flying-car traffic between the towers, restrained grading for a premium feel, powerful natural movement."
                }),
                "width": ("INT", {"default": 1280, "min": 256, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 736, "min": 256, "max": 1920, "step": 32}),
                "duration": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "unet_name": (["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],),
                "clip_name": (["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],),
                "vae_name": (["minimax_h3_video_vae_fp16.safetensors"],),
                "audio_vae": (["minimax_h3_audio_vae_fp32.safetensors"],),
                "turbo_mode": ("BOOLEAN", {"default": True}),
                "lora_name": ([
                    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                    "none"
                ],),
                "turbo_model_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 2.0, "step": 0.05}),
                "turbo_steps": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "modal_url": ("STRING", {"default": MODAL_URL_DEFAULT, "multiline": False}),
                # 勾选后即使参数没变也强制重新调用云端出片
                "force_regenerate": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("VIDEO", "IMAGE", "video_path")
    FUNCTION = "generate_video"
    CATEGORY = "MiniMax H3 (Cloud)"

    def generate_video(
        self, prompt, width, height, duration, noise_seed,
        unet_name, clip_name, vae_name, audio_vae,
        turbo_mode, lora_name, turbo_model_strength, turbo_steps,
        first_frame=None, last_frame=None,
        modal_url=MODAL_URL_DEFAULT,
        force_regenerate=False
    ):
        # 画布：对齐 32 的倍数；低于 H3 原生档位时自动提升（16:9 → 官方 1280×736）
        aligned_width, aligned_height, canvas_note = _native_canvas(width, height)
        actual_steps = turbo_steps if turbo_mode else 20

        # 帧数：24fps，按官方 17k+5 网格吸附
        raw_frames = max(5, round(duration * 24))
        aligned_frames = raw_frames + (5 - raw_frames % 17) % 17
        est_mem_gb = aligned_width * aligned_height * 3 * 4 * aligned_frames / (1024 ** 3)

        print(f"\n=======================================================")
        print(f">>> 🚀 [Modal MiniMax H3 专业版启动]")
        print(f"    分辨率    : {aligned_width} × {aligned_height}" + (f"   ← {canvas_note}" if canvas_note else ""))
        print(f"    目标时长  : {duration} 秒 (渲染帧数: {aligned_frames} 帧 @24fps)")
        print(f"    随机种子  : {noise_seed}")
        print(f"    Turbo加速 : {'开启 (' + str(actual_steps) + ' 步)' if turbo_mode else '关闭 (标准 20 步)'}")
        print(f"    预计内存  : 约 {est_mem_gb:.1f} GB（本地解码 IMAGE 输出占用）")
        print(f"=======================================================\n")

        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted, watermark",
            "seed": noise_seed,
            "steps": actual_steps,
            "width": aligned_width,
            "height": aligned_height,
            "duration": duration,
            "seconds": duration,
            "frames": aligned_frames,
            "num_frames": aligned_frames,
            "length": aligned_frames,
            "turbo_mode": turbo_mode,
            "lora_name": lora_name if turbo_mode else "none",
            "turbo_model_strength": turbo_model_strength,
            "unet_name": unet_name,
            "clip_name": clip_name,
            "vae_name": vae_name,
            "audio_vae": audio_vae,
            "first_frame_b64": tensor_to_base64(first_frame),
            "last_frame_b64": tensor_to_base64(last_frame),
        }

        input_dir = folder_paths.get_input_directory()
        output_filename = f"modal_h3_{noise_seed}.mp4"
        output_path = os.path.join(input_dir, output_filename)
        meta_path = output_path + ".json"
        fingerprint = _payload_fingerprint(payload)

        # ---- 结果复用：参数与上次完全一致时不再调用云端（省一次 GPU 费用）----
        reused = False
        if (REUSE_RESULT and not force_regenerate
                and os.path.isfile(output_path) and os.path.getsize(output_path) > 0):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    cached_fp = json.load(f).get("fingerprint")
            except Exception:
                cached_fp = None
            if cached_fp == fingerprint:
                print(f">>> ♻️  参数与上次完全一致，复用本地已有成片（跳过云端调用，省一次费用）")
                reused = True
            else:
                print(f">>> 本地已有同种子成片但参数不同，将重新出片（现有文件会被覆盖）")

        # ---- 调用云端 ----
        if not reused:
            print(f">>> 正在请求云端（首次冷启动约 3-6 分钟；最长等待 {CLOUD_TIMEOUT} 秒）...")
            response = None
            for attempt in range(1, 4):
                try:
                    response = _post_to_cloud(modal_url, payload, CLOUD_TIMEOUT)
                except Exception as e:
                    if attempt < 3:
                        print(f">>> 第 {attempt} 次请求失败 ({type(e).__name__}: {e})，3 秒后重试...")
                        time.sleep(3)
                        continue
                    raise RuntimeError(
                        f"无法连接云端 {modal_url}：{type(e).__name__}: {e}\n"
                        f"若是代理问题，请检查 v2rayN / Clash Verge 是否正常运行（本节点会自动尝试直连）。"
                    ) from e
                if response.content:
                    break
                if attempt < 3:
                    print(f">>> 第 {attempt} 次请求返回空响应 (HTTP {response.status_code})，3 秒后重试...")
                    time.sleep(3)

            if response.status_code != 200:
                raise RuntimeError(f"云端生成失败: HTTP {response.status_code} - {response.text[:1000]}")
            if not response.content:
                raise RuntimeError(
                    f"云端返回空响应 (HTTP {response.status_code}, "
                    f"Content-Type: {response.headers.get('content-type', '未知')})。"
                    f"这是云端服务问题而非本地解码问题：Modal 函数可能因冷启动或执行超时"
                    f"未写入响应体，请检查 modal.run 服务端日志。"
                )

            video_bytes = _extract_video_bytes(response.content)
            if not _looks_like_video(video_bytes):
                raise RuntimeError(
                    f"云端返回的内容不是有效视频（前 200 字节: {video_bytes[:200]!r}），"
                    f"请检查 modal_url 接口的返回格式。"
                )

            with open(output_path, "wb") as f:
                f.write(video_bytes)
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "fingerprint": fingerprint,
                        "seed": noise_seed,
                        "size": len(video_bytes),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, f, ensure_ascii=False)
            except Exception:
                pass
            print(f">>> 云端出片成功: {output_path} ({len(video_bytes) / 1048576:.1f} MB)")

        # ---- 探测参数 + 本地解码 + 封装 ----
        info = _probe_video(output_path)
        if info:
            print(f">>> 成片参数: {info['width']}×{info['height']} | {info['fps']:.0f}fps | "
                  f"{info['frames']} 帧 | {info['codec']} | 音频流: {info['audio']}"
                  + ("  ⚠️ 无音频" if info["audio"] == 0 else ""))
        else:
            print(f">>> 成片参数: 探测失败（不影响后续解码）")

        print(f">>> 正在本地解码为 IMAGE 张量...")
        images_tensor = decode_video_frames(output_path)
        print(f">>> 解码完成: {tuple(images_tensor.shape)} (帧数×高×宽×通道)")
        print(f">>> 成片完整路径（可直接喂给 Video2X 放大）: {output_path}")

        # 封装为 ComfyUI 官方 VIDEO 对象（带音频，供“获取视频组件”等节点使用）
        fps = info["fps"] if info else 24.0
        video_object = None
        try:
            from comfy_api.latest import InputImpl
            video_object = InputImpl.VideoFromFile(output_path)
        except Exception as e:
            print(f">>> 提示: 未能创建 ComfyUI VIDEO 对象 ({e})，回退为字典格式；"
                  f"如需音频请直接使用 output 里的 video_path 文件")
            video_object = {"images": images_tensor, "fps": fps, "audio": None}

        return (video_object, images_tensor, output_path)


NODE_CLASS_MAPPINGS = {"ModalMiniMaxH3Node": ModalMiniMaxH3Node}
NODE_DISPLAY_NAME_MAPPINGS = {"ModalMiniMaxH3Node": "Image to Video (MiniMax H3) 🚀"}
