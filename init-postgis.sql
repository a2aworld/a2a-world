-- Initialize PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create a sample spatial table for testing
CREATE TABLE IF NOT EXISTS geospatial_data (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    geom GEOMETRY(Point, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO geospatial_data (name, geom) VALUES
('Sample Point 1', ST_GeomFromText('POINT(-122.4194 37.7749)', 4326)),
('Sample Point 2', ST_GeomFromText('POINT(-74.0060 40.7128)', 4326))
ON CONFLICT DO NOTHING;