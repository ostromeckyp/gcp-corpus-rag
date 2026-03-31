.PHONY: dev install test lint clean

install:
	pip install -r requirements.txt

dev:
	uvicorn main:app --reload --port 8080

test:
	pytest tests/ -v

lint:
	flake8 services/ main.py

format:
	black services/ main.py

docker-build:
	docker build -t expense-classifier .

docker-run:
	docker run -p 8080:8080 --env-file .env expense-classifier

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete