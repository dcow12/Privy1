# Privy as a shared web app (Hugging Face Spaces, Render, Railway, Fly.io...).
# Public mode: per-visitor keys and runs, run limits, no local model backend. See app.py.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PRIVY_PUBLIC=1 \
    HOST=0.0.0.0 \
    PORT=7860

# Spaces run containers as uid 1000; give that user the app folder so runs/ is writable.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch: nnsight needs it installed, but models run remotely on NDIF.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY --chown=user . .
# WORKDIR made the folder itself as root; the app writes runs/ here, so hand it to the user.
RUN mkdir -p runs && chown user:user . runs
USER user

EXPOSE 7860
CMD ["python", "app.py"]
