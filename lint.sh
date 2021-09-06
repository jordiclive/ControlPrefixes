#!/usr/bin/env bash

set -uo pipefail
set +e

FAILURE=false

echo "safety"
pipenv run safety check -r requirements.txt -r requirements-dev.txt || FAILURE=true

echo 'black'
pipenv run black src --line-length 120  || FAILURE=true

# echo "pydocstyle"
# pipenv run pydocstyle src || FAILURE=true

#echo "mypy"
#mypy src || FAILURE=true

echo "autoflake"
pipenv run autoflake --remove-all-unused-imports -i -r src || FAILURE=true

#echo "pylint"
#pylint src || FAILURE=true

#echo "bandit"
#bandit -r src || FAILURE=true

echo "isort"
pipenv run isort -rc src || FAILURE=true

##echo "pycodestyle"
##pycodestyle src || FAILURE=true
#
#echo "flake8"
##flake8 src || FAILURE=true

echo "pytest"
pipenv run pytest -s --ignore=tests/test_model.py tests/ || FAILURE=true

echo "training evaluation"
export PYTHONPATH=.
pipenv run python -m unittest

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
