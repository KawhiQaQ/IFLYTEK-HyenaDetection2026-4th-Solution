PYTHON ?= python3
WEIGHTS_ROOT ?= ../weights

.PHONY: verify verify-weights reproduce prepare-data masks train

verify:
	$(PYTHON) tools/verify_package.py

verify-weights:
	$(PYTHON) tools/prepare_weights.py --weights-root $(WEIGHTS_ROOT) --verify-only

reproduce:
	bash scripts/04_rebuild_from_frozen_scores.sh

prepare-data:
	bash scripts/00_prepare_official_data.sh "$(PUBLIC_DATA_ZIP)" "$(HYENA_DATA_ZIP)"

masks:
	bash scripts/01_generate_sam2_masks.sh

train:
	bash scripts/03_train_infer_full.sh
