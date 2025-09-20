# AI Puzzle Pieces Data Pipeline

This directory contains the PostgreSQL/PostGIS data pipeline for the Terra Constellata project. The pipeline is designed to ingest, process, and query geospatial puzzle pieces data.

## Overview

The pipeline consists of several modules:

- `connection.py`: Database connection management
- `schema.py`: Database schema creation and management
- `ingestion.py`: CSV data ingestion with validation and cleaning
- `data_processing.py`: Data cleaning and validation utilities
- `queries.py`: Geospatial query operations
- `pipeline.py`: Main pipeline orchestration script

## Features

- **PostgreSQL/PostGIS Integration**: Full geospatial database support
- **CSV Data Ingestion**: Robust ingestion with error handling
- **Data Validation**: Comprehensive validation of data quality
- **Geospatial Queries**: Advanced spatial query capabilities
- **Data Cleaning**: Automated data cleaning and normalization
- **Batch Processing**: Efficient batch insertion for large datasets
- **Error Handling**: Graceful error handling and logging

## Database Schema

The main table `puzzle_pieces` has the following structure:

```sql
CREATE TABLE puzzle_pieces (
    id SERIAL PRIMARY KEY,
    row_number INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    entity VARCHAR(255),
    sub_entity VARCHAR(255),
    description TEXT,
    source_url TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geom GEOMETRY(POINT, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Installation

1. Ensure PostgreSQL with PostGIS is installed (see main INSTALL.md)
2. Install required Python packages:
   ```bash
   pip install psycopg2-binary geopy pandas
   ```

## Usage

### Command Line Interface

The main pipeline script provides a command-line interface:

```bash
# Initialize the database
python pipeline.py init

# Ingest data from CSV
python pipeline.py ingest --csv path/to/data.csv

# Get database statistics
python pipeline.py stats

# Perform queries (see examples below)
python pipeline.py query --query-type bbox --min-lon -180 --min-lat -90 --max-lon 180 --max-lat 90
```

### Python API

```python
from data.postgis.pipeline import PuzzlePiecesPipeline

# Create pipeline instance
pipeline = PuzzlePiecesPipeline()

# Initialize database
pipeline.initialize()

# Ingest CSV data
result = pipeline.ingest_csv('data.csv')
print(f"Inserted {result['records_inserted']} records")

# Perform geospatial queries
results = pipeline.perform_query('bbox',
                                min_lon=-180, min_lat=-90,
                                max_lon=180, max_lat=90)

# Get statistics
stats = pipeline.get_statistics()

# Clean up
pipeline.cleanup()
```

## Query Types

### Bounding Box Query
```python
results = pipeline.perform_query('bbox',
                                min_lon=-74.0, min_lat=40.7,
                                max_lon=-73.9, max_lat=40.8)
```

### Nearby Points Query
```python
results = pipeline.perform_query('nearby',
                                lon=-74.0, lat=40.7,
                                distance_meters=1000)
```

### Entity-based Queries
```python
# By entity type
results = pipeline.perform_query('by_entity', entity='city')

# By sub-entity type
results = pipeline.perform_query('by_sub_entity', sub_entity='landmark')
```

### Search Queries
```python
# Search by name
results = pipeline.perform_query('search_name',
                                search_term='New York',
                                limit=10)
```

### Spatial Analysis
```python
# Cluster analysis
clusters = pipeline.perform_query('clusters', cluster_distance=1000)

# Nearest neighbors
neighbors = pipeline.perform_query('nearest',
                                  lon=-74.0, lat=40.7,
                                  k=5)
```

## Data Format

CSV files should have the following columns:

- `row_number`: Unique identifier (integer)
- `name`: Name of the puzzle piece (string)
- `entity`: Entity type (string, optional)
- `sub_entity`: Sub-entity type (string, optional)
- `description`: Description (text, optional)
- `source_url`: Source URL (string, optional)
- `latitude`: Latitude coordinate (float, optional)
- `longitude`: Longitude coordinate (float, optional)

Example CSV:
```csv
row_number,name,entity,sub_entity,description,source_url,latitude,longitude
1,Central Park,city,park,Central Park in New York City,https://example.com,40.7829,-73.9654
2,Eiffel Tower,landmark,tower,Iconic tower in Paris,https://example.com,48.8584,2.2945
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **Data Validation**: Detailed validation with error reporting
- **Batch Processing**: Failed batches are logged and can be retried
- **Geospatial Errors**: Invalid coordinates are flagged and handled
- **URL Validation**: Malformed URLs are detected and cleaned

## Logging

The pipeline uses Python's logging module with configurable levels:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Log levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information about pipeline operations
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures

## Performance Considerations

- **Batch Processing**: Large datasets are processed in configurable batches
- **Indexing**: Automatic creation of spatial and regular indexes
- **Connection Pooling**: Efficient database connection management
- **Memory Management**: Streaming processing for large files

## Testing

Run the pipeline with test data:

```bash
# Create test database
python pipeline.py init

# Ingest test data
python pipeline.py ingest --csv test_data.csv

# Run queries
python pipeline.py stats
```

## Integration with Terra Constellata

This pipeline integrates with the existing Terra Constellata project structure:

- Located in `data/postgis/` directory
- Follows the same patterns as the existing `data/ckg/` module
- Uses similar connection and schema management approaches
- Compatible with the project's logging and error handling standards

## Dependencies

- `psycopg2-binary`: PostgreSQL database adapter
- `geopy`: Geocoding and coordinate validation
- `pandas`: Data manipulation and CSV processing
- `python-dateutil`: Date/time handling

## Contributing

When extending the pipeline:

1. Follow the existing code patterns and structure
2. Add comprehensive error handling
3. Include logging for all operations
4. Update this README with new features
5. Add tests for new functionality

## Troubleshooting

### Common Issues

1. **Connection Failed**: Check PostgreSQL is running and credentials are correct
2. **PostGIS Not Available**: Ensure PostGIS extension is installed and enabled
3. **Invalid Coordinates**: Check latitude/longitude ranges (-90 to 90, -180 to 180)
4. **Memory Issues**: Use batch processing for large datasets
5. **Encoding Errors**: Ensure CSV files are UTF-8 encoded

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python pipeline.py --log-level DEBUG ingest --csv data.csv