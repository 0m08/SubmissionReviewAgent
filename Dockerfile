FROM python:3.11

WORKDIR /streamlit_app

# Install core utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip git xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy fixed system package list
COPY packages.txt .

# Install Debian packages safely
RUN apt-get update && \
    xargs -r -a packages.txt apt-get install -y --no-install-recommends || true && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its dependencies
RUN pip install --no-cache-dir playwright && \
    playwright install --with-deps

# Copy your app code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
