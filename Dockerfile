FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY . .
RUN uv sync --frozen
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["uv", "run", "src/app.py"]