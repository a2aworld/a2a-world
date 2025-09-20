from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class UserBase(BaseModel):
    username: str
    email: str


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ContentBase(BaseModel):
    title: str
    body: str
    cat_score: Optional[float] = 0.0


class ContentCreate(ContentBase):
    creator_id: int
    tag_ids: Optional[List[int]] = []


class ContentUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    cat_score: Optional[float] = None
    is_published: Optional[bool] = None


class Content(ContentBase):
    id: int
    is_published: bool
    created_at: datetime
    updated_at: Optional[datetime]
    creator: User
    tags: List["Tag"] = []

    class Config:
        from_attributes = True


class MultimediaBase(BaseModel):
    filename: str
    file_type: str
    metadata: Optional[dict] = {}


class MultimediaCreate(MultimediaBase):
    content_id: int


class Multimedia(MultimediaBase):
    id: int
    file_path: str

    class Config:
        from_attributes = True


class TagBase(BaseModel):
    name: str


class TagCreate(TagBase):
    pass


class Tag(TagBase):
    id: int

    class Config:
        from_attributes = True


class MapBase(BaseModel):
    name: str
    geo_data: dict


class MapCreate(MapBase):
    content_id: int


class Map(MapBase):
    id: int

    class Config:
        from_attributes = True


class ArtworkBase(BaseModel):
    title: str
    description: Optional[str] = None
    generated_by: str


class ArtworkCreate(ArtworkBase):
    multimedia_id: int


class Artwork(ArtworkBase):
    id: int

    class Config:
        from_attributes = True
