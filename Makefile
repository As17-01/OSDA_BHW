format:
	poetry run isort --sl src/
	poetry run black src/
	poetry run flake8 src/ --ignore=D101,D102
