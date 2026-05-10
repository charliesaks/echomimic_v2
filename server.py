"""
FastAPI server wrapping EchoMimicV2 for half-body avatar animation.
Mirrors the interface of the SadTalker server.py so Obake can use either
interchangeably:  POST /generate  →  multipart(image, audio)  →  video/mp4

Start with:
    uvicorn server:app --host 0.0.0.0 --port 8001

Set ECHOMIMIC_BASE_URL=http://localhost:8001 in Obake's .env.local.
"""

import os
import random
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from moviepy.editor import VideoFileClip, AudioFileClip

ECHOMIMIC_DIR    = Path(__file__).parent
WEIGHTS_DIR      = ECHOMIMIC_DIR / "pretrained_weights"
DEFAULT_POSE_DIR = ECHOMIMIC_DIR / "assets" / "halfbody_demo" / "pose" / "01"

pipe         = None
weight_dtype = None
device_str   = None


def _load_pipeline():
    from diffusers import AutoencoderKL, DDIMScheduler
    from omegaconf import OmegaConf

    from src.models.unet_2d_condition           import UNet2DConditionModel
    from src.models.unet_3d_emo                 import EMOUNet3DConditionModel
    from src.models.pose_encoder                import PoseEncoder
    from src.models.whisper.audio2feature       import load_audio_model
    from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline

    dev   = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32

    infer_config = OmegaConf.load(str(ECHOMIMIC_DIR / "configs" / "inference" / "inference_v2.yaml"))

    vae = AutoencoderKL.from_pretrained(str(WEIGHTS_DIR / "sd-vae-ft-mse")).to(device=dev, dtype=dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        str(WEIGHTS_DIR / "sd-image-variations-diffusers"),
        subfolder="unet",
    ).to(dtype=dtype, device=dev)
    reference_unet.load_state_dict(
        torch.load(str(WEIGHTS_DIR / "reference_unet.pth"), map_location="cpu")
    )

    denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
        str(WEIGHTS_DIR / "sd-image-variations-diffusers"),
        str(WEIGHTS_DIR / "motion_module_acc.pth"),
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=dtype, device=dev)
    denoising_unet.load_state_dict(
        torch.load(str(WEIGHTS_DIR / "denoising_unet_acc.pth"), map_location="cpu"),
        strict=False,
    )

    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
        dtype=dtype, device=dev
    )
    pose_net.load_state_dict(torch.load(str(WEIGHTS_DIR / "pose_encoder.pth"), map_location="cpu"))

    audio_processor = load_audio_model(
        model_path=str(WEIGHTS_DIR / "audio_processor" / "tiny.pt"), device=dev
    )

    scheduler = DDIMScheduler(**dict(infer_config.noise_scheduler_kwargs))

    pipeline = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    ).to(device=dev, dtype=dtype)

    return pipeline, dtype, dev


def _load_poses(pose_dir: Path, video_length: int, w: int, h: int, dtype, dev: str) -> torch.Tensor:
    from src.utils.dwpose_util import draw_pose_select_v2

    npy_files = sorted(pose_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if not npy_files:
        raise ValueError(f"No .npy pose files found in {pose_dir}")

    pose_list = []
    for i in range(video_length):
        f        = npy_files[i % len(npy_files)]
        detected = np.load(str(f), allow_pickle=True).tolist()
        imh, imw, rb, re, cb, ce = detected["draw_pose_params"]

        # Render at the original canvas size embedded in the pose data,
        # then resize to the target resolution so any ECHOMIMIC_SIZE works.
        canvas   = np.zeros((imh, imw, 3), dtype=np.uint8)
        rendered = draw_pose_select_v2(detected, imh, imw, ref_w=800)
        canvas[rb:re, cb:ce, :] = np.transpose(np.array(rendered), (1, 2, 0))
        if imh != h or imw != w:
            canvas = np.array(Image.fromarray(canvas).resize((w, h), Image.LANCZOS))

        pose_list.append(
            torch.tensor(canvas, dtype=dtype, device=dev).permute(2, 0, 1) / 255.0
        )

    return torch.stack(pose_list, dim=1).unsqueeze(0)  # [1, C, T, H, W]


def _audio_duration(wav_path: Path) -> float:
    import wave
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 5.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, weight_dtype, device_str
    print("[echomimic] Loading pipeline…")
    pipe, weight_dtype, device_str = _load_pipeline()
    print("[echomimic] Pipeline ready.")
    yield
    pipe = None


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(image: UploadFile = File(...), audio: UploadFile = File(...)):
    if pipe is None:
        raise HTTPException(503, "Pipeline not loaded")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        img_path = tmp / ("avatar" + Path(image.filename or "x.png").suffix)
        aud_path = tmp / ("speech"  + Path(audio.filename or "x.mp3").suffix)
        img_path.write_bytes(await image.read())
        aud_path.write_bytes(await audio.read())

        # EchoMimicV2 requires WAV at 16 kHz mono
        wav_path = tmp / "speech.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aud_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True, capture_output=True,
        )

        fps            = int(os.getenv("ECHOMIMIC_FPS",            "24"))
        # Max frames generated per inference pass.  Keeping this ≤ 48 prevents
        # the 768→48×48 cross-attention from allocating >10 GB on MPS/CPU.
        # The generated clip is looped to cover the full audio duration.
        max_gen_frames = int(os.getenv("ECHOMIMIC_MAX_GEN_FRAMES", "48"))
        # Output size. 512 keeps cross-attention tensors at ~400 MB/layer on MPS;
        # 768 requires ~2 GB/layer and will OOM on most Macs.
        size           = int(os.getenv("ECHOMIMIC_SIZE",            "512"))
        steps          = int(os.getenv("ECHOMIMIC_STEPS",           "6"))
        guidance       = float(os.getenv("ECHOMIMIC_GUIDANCE",      "1.0"))
        ctx_frames     = int(os.getenv("ECHOMIMIC_CONTEXT_FRAMES",  "12"))
        ctx_overlap    = int(os.getenv("ECHOMIMIC_CONTEXT_OVERLAP", "3"))
        w = h          = size

        pose_dir   = DEFAULT_POSE_DIR
        n_poses    = len(list(pose_dir.glob("*.npy")))
        audio_dur  = _audio_duration(wav_path)
        # Only generate up to max_gen_frames; we'll loop the result later.
        L          = min(max_gen_frames, int(audio_dur * fps), n_poses)

        poses     = _load_poses(pose_dir, L, w, h, weight_dtype, device_str)
        ref_image = Image.open(img_path).convert("RGB")
        generator = torch.manual_seed(int(os.getenv("ECHOMIMIC_SEED", str(random.randint(100, 1_000_000)))))

        video = pipe(
            ref_image,
            str(wav_path),
            poses[:, :, :L, ...],
            w, h, L,
            steps,
            guidance,
            generator=generator,
            audio_sample_rate=16000,
            context_frames=ctx_frames,
            fps=fps,
            context_overlap=ctx_overlap,
            start_idx=0,
        ).videos  # [1, C, T, H, W]

        # Loop the generated frames to cover the full audio duration so the
        # output clip is always the correct length for ffmpeg muxing.
        target_L = max(L, int(audio_dur * fps) + 1)
        if target_L > L:
            repeats = (target_L + L - 1) // L
            video   = video.repeat(1, 1, repeats, 1, 1)[:, :, :target_L]
            L       = target_L

        from src.utils.util import save_videos_grid
        silent_path = str(tmp / "silent.mp4")
        save_videos_grid(video[:, :, :L], silent_path, n_rows=1, fps=fps)

        out_path   = tmp / "output.mp4"
        audio_clip = AudioFileClip(str(wav_path)).set_duration(L / fps)
        video_clip = VideoFileClip(silent_path).set_audio(audio_clip)
        video_clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", logger=None)

        return Response(content=out_path.read_bytes(), media_type="video/mp4")
