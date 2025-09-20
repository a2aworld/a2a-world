# Start from the official PostGIS image
FROM postgis/postgis:15-3.4

# Install Python, pip, and required libraries for the seeding script
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install pandas psycopg2-binary && \
    rm -rf /var/lib/apt/lists/*