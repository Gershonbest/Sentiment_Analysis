FROM python:3.12-slim as python
ENV PYTHONUNBUFFERED True

# Install Poetry
ENV POETRY_VERSION=1.8.3
ENV POETRY_VIRTUALENVS_IN_PROJECT=false

RUN pip install --upgrade pip
RUN pip install --no-cache-dir poetry==$POETRY_VERSION
# Configure Poetry to not use virtual environments
RUN poetry config virtualenvs.create false

WORKDIR /code
COPY . /code/
# Install dependencies using Poetry
RUN poetry install --no-interaction --no-ansi --no-cache --no-root --only main


EXPOSE 8000
# CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python3", "sentiment.py"]