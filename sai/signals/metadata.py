"""Metadata signal: EXIF/IPTC fingerprint analysis.

AI-generated images and platform-re-encoded images carry metadata fingerprints
that pixel-level signals cannot detect. This signal examines:

1. **FBMD fingerprint**: Facebook/Meta Binary Metadata in IPTC SpecialInstructions.
   Present when an image has been uploaded to Instagram/Facebook. While not
   proof of AI generation by itself, it proves the image passed through a
   platform pipeline that strips original EXIF and re-encodes.

2. **EXIF absence**: Real photographs from cameras or phones carry EXIF metadata
   (camera make, model, GPS, timestamp, exposure settings). AI-generated images
   have NO camera EXIF. A complete absence of camera metadata is a strong
   AI-generation signal.

3. **Software tag**: Some AI tools (Midjourney, Stable Diffusion WebUI) embed
   software tags like "Midjourney" or "Stable Diffusion" in EXIF/IPTC/XMP.
   This is a deterministic AI fingerprint.

4. **Progressive JPEG**: Instagram and many AI pipelines produce progressive
   JPEGs. Real camera JPEGs are typically baseline. Combined with FBMD, this
   confirms a platform re-encoding pipeline.

This signal requires the image file path (not just the pixel array) because
metadata is stored in file headers, not pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from sai.signals.base import Signal, SignalResult


class MetadataSignal(Signal):
    name = "metadata"

    def __init__(self, file_path: Optional[str | Path] = None) -> None:
        self.file_path = Path(file_path) if file_path else None

    def _check_fbmd(self, img: Image.Image) -> bool:
        """Check for Facebook/Meta Binary Metadata in IPTC SpecialInstructions."""
        try:
            if hasattr(img, "applist"):
                for marker, data in img.applist:
                    if marker == "APP14":
                        if b"FBMD" in data:
                            return True
            # Also check raw info dict
            if "photoshop" in img.info:
                raw = img.info.get("photoshop", b"")
                if isinstance(raw, (bytes, bytearray)) and b"FBMD" in raw:
                    return True
        except Exception:
            pass
        return False

    def _check_exif_camera(self, img: Image.Image) -> bool:
        """Check if EXIF contains camera metadata (make, model, lens, etc.)."""
        try:
            exif = img.getexif()
            if not exif:
                return False
            camera_tags = {
                271,  # Make
                272,  # Model
                33434,  # ExposureTime
                33437,  # FNumber
                37500,  # MakerNote
                37386,  # FocalLength
                37378,  # ApertureValue
                37377,  # ShutterSpeedValue
                37383,  # MeteringMode
                37384,  # LightSource
                37385,  # Flash
                41988,  # FocalLengthIn35mmFilm
            }
            for tag_id in camera_tags:
                if tag_id in exif:
                    val = exif[tag_id]
                    if val and str(val).strip():
                        return True
        except Exception:
            pass
        return False

    def _check_c2pa(self, file_path: str | Path) -> dict:
        """Inspect file binary headers for C2PA / Coalition for Content Provenance metadata.

        C2PA is the international standard adopted by Google, OpenAI, Adobe, Microsoft,
        Sony, Leica, and Nikon to cryptographically sign content origin.
        """
        try:
            p = Path(file_path)
            if not p.is_file():
                return {"has_c2pa": False, "is_synthetic": False, "is_capture": False}
            with open(p, "rb") as f:
                # Read initial segment containing metadata manifests/headers
                header = f.read(131072)
            has_c2pa_marker = (
                b"c2pa" in header
                or b"C2PA" in header
                or b"urn:c2pa" in header
                or b"jumbf" in header
            )
            if not has_c2pa_marker:
                return {"has_c2pa": False, "is_synthetic": False, "is_capture": False}

            synthetic_terms = [
                b"trainedAlgorithmicMedia",
                b"compositeSynthetic",
                b"c2pa.synthetic",
                b"c2pa.created",
                b"c2pa.ai",
                b"c2pa.data_mining",
                b"algorithmicMedia",
                b"syntheticMedia",
            ]
            is_synthetic = any(term in header for term in synthetic_terms)
            is_capture = (
                b"c2pa.capture" in header or b"c2pa.placed" in header
            ) and not is_synthetic

            return {
                "has_c2pa": True,
                "is_synthetic": is_synthetic,
                "is_capture": is_capture,
            }
        except Exception:
            return {"has_c2pa": False, "is_synthetic": False, "is_capture": False}

    def _check_software_tag(self, img: Image.Image) -> Optional[str]:
        """Check for AI/SI tool signatures in EXIF/IPTC software tags."""
        ai_si_signatures = [
            # Traditional AI generator signatures
            "midjourney", "stable diffusion", "dall-e", "dalle",
            "flux", "novelai", "niji", "comfyui", "automatic1111",
            "dreamstudio", "leonardo", "firefly", "imagen",
            "gpt-4", "chatgpt", "bing image", "craiyon",
            # Modern SI (Synthetic Intelligence / Synthetic Media) & Frontier models
            "synthetic intelligence", "synthetic media", "synthetic image",
            "synthid", "c2pa.synthetic", "trainedalgorithmicmedia",
            "black forest labs", "sd3", "stable diffusion 3",
            "ideogram", "recraft", "kling", "luma", "runway", "gen-3",
            "grok-2", "aurora", "seedream", "cogview", "minimax", "hailuo",
        ]
        try:
            exif = img.getexif()
            # Check Software tag (305)
            software = str(exif.get(305, "")).lower()
            for sig in ai_si_signatures:
                if sig in software:
                    return sig
            # Check IPTC Object Name (2:5) and Caption (2:120)
            # PIL doesn't expose IPTC directly, check info dict
            for key, val in img.info.items():
                if isinstance(val, (str, bytes)):
                    val_lower = str(val).lower()
                    for sig in ai_si_signatures:
                        if sig in val_lower:
                            return sig
        except Exception:
            pass
        return None

    def _check_progressive(self, img: Image.Image) -> bool:
        """Check if JPEG is progressive (common in AI/platform-re-encoded images)."""
        return bool(img.info.get("progressive", False))

    def _check_jfif_only(self, img: Image.Image) -> bool:
        """Check if image has only JFIF metadata (no EXIF, no camera data).
        JFIF-only images are typically generated/synthetic, not from cameras."""
        has_exif = bool(img.getexif())
        has_jfif = "jfif" in img.info
        return has_jfif and not has_exif

    def analyze(self, image: np.ndarray) -> SignalResult:
        """Analyze pixel array - limited analysis without file path."""
        # Without file path, we can only do pixel-level heuristics
        # Return neutral score with low weight
        return SignalResult(
            score=0.5,
            weight=0.1,
            features={"note": "no file path provided, metadata analysis skipped"},
        )

    def analyze_file(self, file_path: str | Path) -> SignalResult:
        """Analyze image file metadata including EXIF, IPTC, and C2PA provenance."""
        img = Image.open(file_path)

        c2pa_info = self._check_c2pa(file_path)
        has_fbmd = self._check_fbmd(img)
        has_camera = self._check_exif_camera(img)
        ai_software = self._check_software_tag(img)
        is_progressive = self._check_progressive(img)
        jfif_only = self._check_jfif_only(img)

        # 1. C2PA Signed Provenance (Cryptographic ground truth)
        if c2pa_info["has_c2pa"] and c2pa_info["is_synthetic"]:
            return SignalResult(
                score=1.0,
                weight=1.0,
                features={
                    "c2pa": c2pa_info,
                    "has_c2pa": True,
                    "is_c2pa_synthetic": True,
                    "ai_software": "c2pa_synthetic_manifest",
                    "has_fbmd": has_fbmd,
                    "has_camera_exif": has_camera,
                    "is_progressive": is_progressive,
                    "jfif_only": jfif_only,
                    "deterministic": True,
                },
            )

        if c2pa_info["has_c2pa"] and c2pa_info["is_capture"]:
            return SignalResult(
                score=0.05,
                weight=1.0,
                features={
                    "c2pa": c2pa_info,
                    "has_c2pa": True,
                    "is_c2pa_capture": True,
                    "ai_software": None,
                    "has_fbmd": has_fbmd,
                    "has_camera_exif": True,
                    "is_progressive": is_progressive,
                    "jfif_only": jfif_only,
                    "deterministic": True,
                },
            )

        # 2. Explicit AI / SI Software tag embedded in EXIF/IPTC
        if ai_software:
            return SignalResult(
                score=1.0,
                weight=1.0,
                features={
                    "c2pa": c2pa_info,
                    "ai_software": ai_software,
                    "has_fbmd": has_fbmd,
                    "has_camera_exif": has_camera,
                    "is_progressive": is_progressive,
                    "jfif_only": jfif_only,
                    "deterministic": True,
                },
            )

        if has_camera:
            # Real camera metadata present - strong real signal
            return SignalResult(
                score=0.1,
                weight=0.9,
                features={
                    "c2pa": c2pa_info,
                    "has_fbmd": has_fbmd,
                    "has_camera_exif": True,
                    "is_progressive": is_progressive,
                    "jfif_only": False,
                    "deterministic": False,
                },
            )

        # No camera EXIF - likely synthetic or platform-re-encoded
        score = 0.5
        weight = 0.5

        if jfif_only:
            score = 0.85
            weight = 0.85
        elif is_progressive and not has_camera:
            score = 0.70
            weight = 0.65
        elif not has_camera:
            score = 0.65
            weight = 0.55

        # FBMD alone doesn't prove AI (real photos uploaded to IG also get it)
        # But FBMD + no camera EXIF = original was not a camera photo
        if has_fbmd and not has_camera:
            score = min(score + 0.1, 0.95)
            weight = min(weight + 0.1, 0.9)

        return SignalResult(
            score=float(np.clip(score, 0.0, 1.0)),
            weight=float(np.clip(weight, 0.0, 1.0)),
            features={
                "c2pa": c2pa_info,
                "has_fbmd": has_fbmd,
                "has_camera_exif": False,
                "ai_software": None,
                "is_progressive": is_progressive,
                "jfif_only": jfif_only,
                "deterministic": False,
            },
        )
