from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    contents = relationship("Content", back_populates="creator")


class Content(Base):
    __tablename__ = "contents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    body = Column(Text)
    cat_score = Column(Float, default=0.0)
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"))

    creator = relationship("User", back_populates="contents")
    multimedia = relationship("Multimedia", back_populates="content")
    tags = relationship("Tag", secondary="content_tags", back_populates="contents")


class Multimedia(Base):
    __tablename__ = "multimedia"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)  # image, video, audio, etc.
    content_id = Column(Integer, ForeignKey("contents.id"))
    metadata = Column(JSON)  # For additional info like dimensions, duration, etc.

    content = relationship("Content", back_populates="multimedia")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    contents = relationship("Content", secondary="content_tags", back_populates="tags")


class ContentTag(Base):
    __tablename__ = "content_tags"

    content_id = Column(Integer, ForeignKey("contents.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)


class Map(Base):
    __tablename__ = "maps"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    geo_data = Column(JSON)  # GeoJSON or similar
    content_id = Column(Integer, ForeignKey("contents.id"))

    content = relationship("Content")


class Artwork(Base):
    __tablename__ = "artworks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    generated_by = Column(String)  # AI model or tool used
    multimedia_id = Column(Integer, ForeignKey("multimedia.id"))

    multimedia = relationship("Multimedia")
