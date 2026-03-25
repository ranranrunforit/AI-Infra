# cloud/functions/mcap_indexer/Dockerfile
# Builds the MCAP indexer Cloud Run container

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FIX #5: Both source files must be copied.
# Previously only main.py was copied, so vlm_tagger import silently failed at
# runtime, VLM_AVAILABLE was set to False, and VLM tagging was disabled with
# no error message whatsoever.
COPY main.py vlm_tagger.py ./

ENV PORT=8080
EXPOSE 8080

CMD ["functions-framework", "--target=index_mcap_file", "--signature-type=cloudevent", "--port=8080"]
