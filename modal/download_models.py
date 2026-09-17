"""把 MiniMax H3 相关模型从 HuggingFace 下载进 Modal Volume。

用法：
    # 先把下面的 VOLUME_NAME 改成你自己的卷名
    modal run modal/download_models.py

说明：
  - 权重总量约 26GB（UNet int8 ~9GB + Qwen3VL-32B 文本编码器 ~17GB + VAE/LoRA）
  - 断点友好：已存在的文件会跳过，可重复执行
  - 下载完成后会 volume.commit() 持久化，端点即可直接挂载使用
"""
import os

import modal

VOLUME_NAME = os.environ.get("H3_VOLUME_NAME", "h3-model-volume")
HF_REPO_MAIN = "Comfy-Org/MiniMax-H3"
HF_REPO_LORA = "lightx2v/Minimax-h3-Turbo"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# (HF 仓库, 仓库内路径)
FILES = [
    # UNet（int8 量化，官方体积最小可用档；nvfp4 需 Blackwell，A100 不可用）
    (HF_REPO_MAIN, "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    # 文本/视觉编码器 Qwen3VL-32B
    (HF_REPO_MAIN, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    # VAE：视频 fp16 + 音频 fp32（原生立体声依赖音频 VAE）
    (HF_REPO_MAIN, "vae/minimax_h3_video_vae_fp16.safetensors"),
    (HF_REPO_MAIN, "vae/minimax_h3_audio_vae_fp32.safetensors"),
    # Turbo LoRA：8 步加速（缺它端点会回退到 20 步，采样耗时约 2.5 倍）
    (HF_REPO_LORA, "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
]


@app.function(image=image, volumes={"/models": volume}, timeout=3600)
def download():
    from huggingface_hub import hf_hub_download

    for repo_id, file_path in FILES:
        dest = os.path.join("/models", file_path)
        if os.path.exists(dest):
            print(">>> [已存在跳过] %s" % file_path)
            continue
        print(">>> 正在下载: %s  (%s)" % (file_path, repo_id))
        hf_hub_download(repo_id=repo_id, filename=file_path, local_dir="/models")
        print(">>> 下载完成: %s" % file_path)

    volume.commit()
    print(">>> 全部模型已写入 Modal Volume: %s" % VOLUME_NAME)


@app.local_entrypoint()
def main():
    download.remote()
