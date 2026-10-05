from fastapi import APIRouter, UploadFile

from models.schemas import ProfileReport
from services.profiler import profile_dataset
from utils.io import UnsupportedFileError, read_tabular
from utils.problem import APIError

router = APIRouter(prefix="/api", tags=["profile"])


@router.post("/profile", response_model=ProfileReport)
async def profile_upload(file: UploadFile) -> ProfileReport:
    content = await file.read()
    try:
        df = read_tabular(file.filename or "", content)
        return profile_dataset(df)
    except UnsupportedFileError as exc:
        raise APIError(400, "不支持的文件类型", str(exc)) from exc
    except ValueError as exc:
        raise APIError(422, "数据无法解析", str(exc)) from exc
