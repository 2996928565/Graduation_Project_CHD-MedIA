from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.auth import require_admin
from config.settings import settings
from db.database import get_db
from db.models import NormalityModel, User
from loguru import logger

router = APIRouter(prefix="/normality", tags=["常模训练"])

_RUNS: Dict[str, Dict[str, Any]] = {}


class NormalityTrainStartResponse(BaseModel):
    run_id: str
    status: str
    started_at: str
    pred_dir: str
    output_model: str
    log_path: str


class NormalityTrainStatusResponse(BaseModel):
    run_id: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    pred_dir: str
    output_model: str
    summary_path: Optional[str] = None
    log_path: str
    model_id: Optional[int] = None
    message: Optional[str] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runs_root() -> Path:
    base = Path(settings.upload_dir)
    if not base.is_absolute():
        base = (_project_root() / base).resolve()
    root = (base / "normality_train_runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _models_root() -> Path:
    base = Path(settings.upload_dir)
    if not base.is_absolute():
        base = (_project_root() / base).resolve()
    root = (base / "normality_models").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _doctor_label(user: User) -> str:
    return ((user.full_name or "").strip() or (user.username or "").strip())

def _sanitize_model_filename(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raw = "mri_normal_heart_mlp"

    invalid = '<>:"/\\|?*\n\r\t'
    out = "".join(ch for ch in raw if ch not in invalid and ord(ch) >= 32).strip()
    out = out.replace(" ", "_")
    if not out:
        out = "mri_normal_heart_mlp"

    if out.lower().endswith(".pth"):
        out = out[: -len(".pth")]
    out = out.strip(" .")
    if not out:
        out = "mri_normal_heart_mlp"

    if len(out) > 80:
        out = out[:80]

    return f"{out}.pth"


def _safe_extract(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.endswith("/"):
                continue
            dest = (target_dir / name).resolve()
            if target_dir.resolve() not in dest.parents and dest != target_dir.resolve():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="压缩包包含非法路径",
                )
        zf.extractall(target_dir)


def _write_run_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    (run_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_run_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_text(path: Path, max_lines: int = 200) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            try:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 200_000), 0)
            except Exception:
                pass
            raw = f.read()
    except Exception:
        return ""
    data = None
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            data = raw.decode(enc, errors="strict")
            break
        except Exception:
            continue
    if data is None:
        data = raw.decode("utf-8", errors="replace")
    lines = data.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _reload_mri_normal_model(model_path: str) -> None:
    settings.mri_normal_model_path = model_path
    try:
        from core.mri.detector import get_mri_detector

        det = get_mri_detector()
        det.normal_model_path = str(model_path)
        det._load_normal_model()
    except Exception as e:
        logger.warning(f"MRI 常模模型热加载失败: {e}")


def _activate_model(db: Session, model: NormalityModel) -> None:
    db.query(NormalityModel).filter(
        NormalityModel.modality == model.modality,
        NormalityModel.model_type == model.model_type,
        NormalityModel.is_active == True,
    ).update({"is_active": False})
    model.is_active = True
    db.add(model)
    db.commit()
    _reload_mri_normal_model(model.model_path)


def _update_run_status(run: Dict[str, Any]) -> Dict[str, Any]:
    proc = run.get("process")
    if proc is None:
        return run
    if run.get("status") in {"succeeded", "failed"}:
        return run
    code = proc.poll()
    if code is None:
        run["status"] = "running"
        return run
    run["exit_code"] = int(code)
    run["finished_at"] = datetime.now().isoformat()
    if int(code) == 0:
        run["status"] = "succeeded"
    else:
        run["status"] = "failed"
    run_dir = Path(run["run_dir"])
    summary_path = Path(run["output_model"]).with_suffix(".train_summary.json")
    if summary_path.exists():
        run["summary_path"] = str(summary_path)
    meta = {k: v for k, v in run.items() if k not in {"process"}}
    _write_run_meta(run_dir, meta)
    return run


def _register_model_if_needed(db: Session, run: Dict[str, Any]) -> Optional[int]:
    if run.get("status") != "succeeded":
        return None
    if run.get("model_id"):
        return int(run["model_id"])

    output_model = Path(run["output_model"])
    if not output_model.exists():
        return None

    run_id = str(run.get("run_id") or "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = (_models_root() / "mri" / "mlp" / f"{stamp}_{run_id}").resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    chosen_name = str((run.get("params") or {}).get("model_name") or output_model.name)
    model_dst = bundle_dir / _sanitize_model_filename(chosen_name)
    summary_src = Path(run.get("summary_path") or str(output_model.with_suffix(".train_summary.json")))
    log_src = Path(run.get("log_path") or "")

    shutil.copyfile(str(output_model), str(model_dst))
    summary_dst = None
    if summary_src.exists():
        summary_dst = bundle_dir / "train_summary.json"
        shutil.copyfile(str(summary_src), str(summary_dst))
    log_dst = None
    if log_src.exists():
        log_dst = bundle_dir / "train.log"
        shutil.copyfile(str(log_src), str(log_dst))

    display_name = (run.get("display_name") or "").strip()
    if not display_name:
        display_name = str((run.get("params") or {}).get("model_name") or f"MRI 常模 MLP {stamp}")

    model = NormalityModel(
        modality="mri",
        model_type="mlp",
        display_name=display_name,
        run_id=run_id,
        model_path=str(model_dst),
        summary_path=str(summary_dst) if summary_dst else None,
        log_path=str(log_dst) if log_dst else None,
        params=dict(run.get("params") or {}),
        created_by=str(run.get("created_by") or ""),
        is_active=False,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    run["model_id"] = int(model.id)
    meta = {k: v for k, v in run.items() if k not in {"process"}}
    _write_run_meta(Path(run["run_dir"]), meta)
    return int(model.id)


@router.post(
    "/train-mlp",
    summary="上传分割预测标签并训练 MRI 常模模型（MLP AutoEncoder）",
    response_model=NormalityTrainStartResponse,
)
async def train_normality_mlp(
    dataset_zip: UploadFile = File(..., description="zip包，包含 *_prediction.nii.gz，可选 normal_list.txt"),
    model_name: str = Form(default="mri_normal_heart_mlp", description="输出模型文件名（不含路径，可不带.pth后缀）"),
    epochs: int = Form(default=200, ge=20, le=2000),
    hidden_dims: str = Form(default="64,32"),
    latent_dim: int = Form(default=8, ge=1, le=256),
    batch_size: int = Form(default=8, ge=1, le=128),
    lr: float = Form(default=1e-3, gt=0.0, le=1.0),
    weight_decay: float = Form(default=1e-5, ge=0.0, le=1.0),
    threshold_quantile: float = Form(default=0.99, gt=0.0, lt=1.0),
    device: str = Form(default="cuda"),
    auto_activate: bool = Form(default=True),
    pred_is_raw_mmwhs: bool = Form(default=False),
    normal_list_name: str = Form(default="normal_list.txt"),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> NormalityTrainStartResponse:
    filename = (dataset_zip.filename or "").lower()
    if not filename.endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="仅支持上传 .zip 数据集")

    run_id = uuid.uuid4().hex
    run_dir = (_runs_root() / run_id).resolve()
    data_dir = (run_dir / "data").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = run_dir / "dataset.zip"
    content = await dataset_zip.read()
    zip_path.write_bytes(content)

    try:
        _safe_extract(zip_path, data_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="zip文件损坏或格式不正确")

    normal_list_path = data_dir / normal_list_name
    normal_list_arg = str(normal_list_path) if normal_list_path.exists() else ""

    dims: list[int] = []
    for part in str(hidden_dims or "").split(","):
        t = part.strip()
        if not t:
            continue
        try:
            dims.append(int(t))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="hidden_dims 格式错误，应为逗号分隔整数，例如 64,32",
            )
    if not dims:
        dims = [64, 32]
    if any(d <= 0 for d in dims) or len(dims) > 6:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="hidden_dims 需为正整数，层数建议不超过 6",
        )

    dev = (device or "cuda").strip().lower()
    if dev not in {"cuda", "cpu"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="device 仅支持 cuda / cpu",
        )

    output_dir = (run_dir / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = _sanitize_model_filename(model_name)
    output_model = output_dir / output_filename
    log_path = run_dir / "train.log"

    cmd = [
        sys.executable,
        "-u",
        "backend/training/train_normal_heart_mlp.py",
        "--pred_dir",
        str(data_dir),
        "--hidden_dims",
        *[str(x) for x in dims],
        "--latent_dim",
        str(int(latent_dim)),
        "--epochs",
        str(int(epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--weight_decay",
        str(float(weight_decay)),
        "--threshold_quantile",
        str(float(threshold_quantile)),
        "--device",
        str(dev),
        "--output_model",
        str(output_model),
    ]
    if normal_list_arg:
        cmd.extend(["--normal_list", normal_list_arg])
    if pred_is_raw_mmwhs:
        cmd.append("--pred_is_raw_mmwhs")

    started_at = datetime.now().isoformat()
    meta = {
        "run_id": run_id,
        "status": "starting",
        "started_at": started_at,
        "pred_dir": str(data_dir),
        "output_model": str(output_model),
        "summary_path": None,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "finished_at": None,
        "exit_code": None,
        "model_id": None,
        "created_by": _doctor_label(admin_user),
        "params": {
            "model_name": output_filename,
            "epochs": int(epochs),
            "hidden_dims": dims,
            "latent_dim": int(latent_dim),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "threshold_quantile": float(threshold_quantile),
            "device": dev,
            "auto_activate": bool(auto_activate),
            "pred_is_raw_mmwhs": bool(pred_is_raw_mmwhs),
            "normal_list_name": normal_list_name,
        },
        "cmd": cmd,
    }
    _write_run_meta(run_dir, meta)

    import subprocess

    env = dict(**os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_project_root()),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    run = dict(meta)
    run["process"] = proc
    run["status"] = "running"
    _RUNS[run_id] = run

    logger.info(f"常模训练任务已启动 | run_id={run_id} | pid={proc.pid}")
    return NormalityTrainStartResponse(
        run_id=run_id,
        status="running",
        started_at=started_at,
        pred_dir=str(data_dir),
        output_model=str(output_model),
        log_path=str(log_path),
    )


@router.get(
    "/train-mlp/{run_id}",
    summary="查询 MRI 常模训练任务状态（MLP）",
    response_model=NormalityTrainStatusResponse,
)
def get_train_status(
    run_id: str,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
) -> NormalityTrainStatusResponse:
    run = _RUNS.get(run_id)
    if run is None:
        run_dir = (_runs_root() / run_id).resolve()
        if not run_dir.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到训练任务")
        meta = _read_run_meta(run_dir)
        if not meta:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到训练任务元信息")
        return NormalityTrainStatusResponse(**meta)

    run = _update_run_status(run)
    _register_model_if_needed(db, run)
    if run.get("status") == "succeeded" and bool((run.get("params") or {}).get("auto_activate", True)):
        model_id = run.get("model_id")
        if model_id:
            model = db.query(NormalityModel).filter(NormalityModel.id == int(model_id)).first()
            if model and not bool(model.is_active):
                _activate_model(db, model)
    _RUNS[run_id] = run
    payload = {k: v for k, v in run.items() if k != "process"}
    return NormalityTrainStatusResponse(**payload)


@router.get(
    "/train-mlp/{run_id}/log",
    summary="获取 MRI 常模训练日志（末尾）",
)
def get_train_log(
    run_id: str,
    lines: int = Query(default=200, ge=20, le=2000),
    _admin=Depends(require_admin),
) -> Dict[str, Any]:
    run = _RUNS.get(run_id)
    if run is None:
        run_dir = (_runs_root() / run_id).resolve()
        meta = _read_run_meta(run_dir)
        log_path = Path(meta.get("log_path") or "")
    else:
        log_path = Path(run.get("log_path") or "")
    if not log_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到日志文件")
    text = _tail_text(log_path, max_lines=int(lines))
    return {"run_id": run_id, "log": text}


class NormalityModelItem(BaseModel):
    id: int
    modality: str
    model_type: str
    display_name: Optional[str] = None
    run_id: Optional[str] = None
    model_path: str
    summary_path: Optional[str] = None
    log_path: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


@router.get(
    "/models",
    summary="列出已训练的常模模型",
    response_model=list[NormalityModelItem],
)
def list_models(
    modality: str = Query(default="mri"),
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
) -> list[NormalityModelItem]:
    q = db.query(NormalityModel).filter(NormalityModel.modality == (modality or "mri"))
    rows = q.order_by(NormalityModel.is_active.desc(), NormalityModel.created_at.desc()).all()
    return [
        NormalityModelItem(
            id=r.id,
            modality=r.modality,
            model_type=r.model_type,
            display_name=r.display_name,
            run_id=r.run_id,
            model_path=r.model_path,
            summary_path=r.summary_path,
            log_path=r.log_path,
            params=dict(r.params or {}),
            created_by=r.created_by,
            is_active=bool(r.is_active),
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]


@router.post(
    "/models/{model_id}/activate",
    summary="启用某个常模模型（立即生效）",
    response_model=NormalityModelItem,
)
def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
) -> NormalityModelItem:
    model = db.query(NormalityModel).filter(NormalityModel.id == int(model_id)).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="模型不存在")
    if not model.model_path or not Path(model.model_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="模型文件不存在，无法启用")
    _activate_model(db, model)
    return NormalityModelItem(
        id=model.id,
        modality=model.modality,
        model_type=model.model_type,
        display_name=model.display_name,
        run_id=model.run_id,
        model_path=model.model_path,
        summary_path=model.summary_path,
        log_path=model.log_path,
        params=dict(model.params or {}),
        created_by=model.created_by,
        is_active=bool(model.is_active),
        created_at=model.created_at.isoformat() if model.created_at else "",
    )
