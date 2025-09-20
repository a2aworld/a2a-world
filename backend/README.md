# Galactic Storybook CMS

A headless CMS built with FastAPI and SQLAlchemy as an open-source alternative to Strapi, designed for the Terra Constellata project.

## Features

- **Content Management**: CRUD operations for stories and creations
- **Multimedia Support**: File upload and storage for images, videos, and audio
- **CAT-Score Auto-Publishing**: Automatic publishing of high-quality content
- **Interactive Maps**: GeoJSON-based map support
- **Generated Artwork**: Integration with AI-generated content
- **RESTful API**: Full API for frontend integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the CMS:
```bash
python run_cms.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Content
- `GET /api/content/` - List all content
- `POST /api/content/` - Create new content
- `GET /api/content/{id}` - Get content by ID
- `PUT /api/content/{id}` - Update content
- `DELETE /api/content/{id}` - Delete content

### Multimedia
- `POST /api/multimedia/upload` - Upload file
- `GET /api/multimedia/{id}` - Get multimedia info

### Pipeline
- `POST /api/pipeline/auto-publish` - Trigger auto-publishing
- `GET /api/pipeline/high-cat-score` - Get high CAT-score content
- `POST /api/pipeline/publish/{id}` - Publish specific content

### Maps
- `POST /api/maps/` - Create map
- `GET /api/maps/{content_id}` - Get maps for content

### Artworks
- `POST /api/artworks/` - Create artwork
- `GET /api/artworks/{multimedia_id}` - Get artworks for multimedia

## Database Models

- **Content**: Stories and creations with CAT-scores
- **Multimedia**: Files associated with content
- **User**: Content creators
- **Tag**: Content categorization
- **Map**: Interactive maps
- **Artwork**: AI-generated artwork

## Auto-Publishing Pipeline

Content with CAT-score >= 0.8 is automatically published. The pipeline can be triggered manually or run as a background task.