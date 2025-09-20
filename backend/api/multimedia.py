from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
import aiofiles
import os
import re
from ..database import get_db
from .. import crud, schemas

router = APIRouter()

UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.mp3', '.wav', '.pdf', '.txt'}

os.makedirs(UPLOAD_DIR, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and invalid characters."""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace('..', '')
    return filename.strip()


@router.post("/upload", response_model=schemas.Multimedia)
async def upload_file(
    file: UploadFile = File(...), content_id: int = None, db: Session = Depends(get_db)
):
    if not content_id:
        raise HTTPException(status_code=400, detail="content_id is required")

    # Validate file size
    file_size = 0
    content_chunks = []
    while True:
        chunk = await file.read(1024 * 1024)  # Read 1MB chunks
        if not chunk:
            break
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        content_chunks.append(chunk)

    # Validate file extension
    if file.filename:
        _, ext = os.path.splitext(file.filename.lower())
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Sanitize filename
    safe_filename = sanitize_filename(file.filename or "unnamed_file")
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    # Ensure unique filename
    counter = 1
    original_path = file_path
    while os.path.exists(file_path):
        name, ext = os.path.splitext(original_path)
        file_path = f"{name}_{counter}{ext}"
        counter += 1

    # Write file
    async with aiofiles.open(file_path, "wb") as out_file:
        for chunk in content_chunks:
            await out_file.write(chunk)

    # Determine file type
    file_type = file.content_type.split("/")[0] if file.content_type else "unknown"

    multimedia_data = schemas.MultimediaCreate(
        filename=safe_filename, file_type=file_type, content_id=content_id
    )

    return crud.create_multimedia(
        db=db, multimedia=multimedia_data, file_path=file_path
    )


@router.get("/{multimedia_id}", response_model=schemas.Multimedia)
def get_multimedia(multimedia_id: int, db: Session = Depends(get_db)):
    db_multimedia = crud.get_multimedia(db, multimedia_id=multimedia_id)
    if db_multimedia is None:
        raise HTTPException(status_code=404, detail="Multimedia not found")
    return db_multimedia
