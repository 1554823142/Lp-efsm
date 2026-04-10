import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from protocol_infer.visualization.service import VisualizationService


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_DATA_ROOT = os.path.join(_ROOT, "Data")

service = VisualizationService(data_root=_DATA_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="P-EFSM Visualizer", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets() -> dict:
    return {"items": service.list_datasets()}


@app.post("/api/learn")
def learn(
    protocol: str = Form(...),
    data_dir: str = Form(...),
    max_pcaps: int = Form(6),
    max_sessions: int = Form(200),
    profile: str = Form("balanced"),
    test_ratio: float = Form(0.2),
    seed: int = Form(42),
    dataset_mode: str = Form("pcap"),
    synthetic_sessions: int = Form(0),
    synthetic_session_len: int = Form(20),
    prune_mode: str = Form("none"),
    prune_percentile: int = Form(70),
) -> dict:
    try:
        return service.learn_from_dataset(
            protocol=protocol,
            data_dir=data_dir,
            max_pcaps=max_pcaps,
            max_sessions=max_sessions,
            profile=profile,
            test_ratio=test_ratio,
            seed=seed,
            dataset_mode=dataset_mode,
            synthetic_sessions=synthetic_sessions,
            synthetic_session_len=synthetic_session_len,
            prune_mode=prune_mode,
            prune_percentile=prune_percentile,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: str) -> dict:
    try:
        return service.get_artifact(artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/artifacts/{artifact_id}/upload-pcap")
async def upload_pcap(artifact_id: str, file: UploadFile = File(...)) -> dict:
    suffix = os.path.splitext(file.filename or "upload.pcap")[1] or ".pcap"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_path = temp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)
        return service.replay_uploaded_pcap(artifact_id=artifact_id, pcap_path=temp_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
