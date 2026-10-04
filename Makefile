# One-command entry points. Run `make help` for the list.
PY ?= .venv/bin/python

.PHONY: help setup pipeline eda select train predict verify notebooks final test clean

help:            ## show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:           ## create .venv and install requirements.txt
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

pipeline:        ## full pipeline: eda -> select -> train -> predict -> verify (~5 min)
	$(PY) -m fraudlab all

eda:             ## data audit, shift table, README figures
	$(PY) -m fraudlab eda
select:          ## cross-validate the model zoo and pick the final model
	$(PY) -m fraudlab select
train:           ## fit the selected model on all training rows
	$(PY) -m fraudlab train
predict:         ## write results/final/submission.csv
	$(PY) -m fraudlab predict
verify:          ## check the rebuilt model/submission against the shipped model and the 0.19586 upload
	$(PY) -m fraudlab verify

notebooks:       ## re-execute the 3 notebooks in place (~10 min)
	PYTHON=$(PY) scripts/run_notebooks.sh notebooks/*.ipynb

final:           ## execute the final deliverable notebook
	PYTHON=$(PY) scripts/run_notebooks.sh final_notebook/predict_catboost.ipynb

test:            ## unit tests
	$(PY) -m pytest -q

clean:           ## remove caches and pipeline outputs (keeps results/leaderboard_uploads)
	rm -rf results/final results/model_selection results/logs catboost_info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
