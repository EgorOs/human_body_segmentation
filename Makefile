.PHONY: *

PYTHON_EXEC := python3.10
CLEARML_PROJECT_NAME := human_body_segmentation
CLEARML_DATASET_NAME := pascal_parts_dataset

DATASET_TEMP_DIR := ".data_temp"


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python


migrate_dataset: reset_dataset_dir download_pascal_part_annotations download_pascal_voc_2010_dataset remove_extra_images upload_data_to_clearml
	rm -R $(DATASET_TEMP_DIR)


reset_dataset_dir:
	rm -R $(DATASET_TEMP_DIR) || true
	mkdir $(DATASET_TEMP_DIR)


download_pascal_part_annotations:
	wget "http://liangchiehchen.com/data/pascal_person_part.zip" -O $(DATASET_TEMP_DIR)/pascal_person_part.zip
	unzip -q $(DATASET_TEMP_DIR)/pascal_person_part.zip -d $(DATASET_TEMP_DIR)
	rm $(DATASET_TEMP_DIR)/pascal_person_part.zip
	find $(DATASET_TEMP_DIR) -type f -name '.DS_Store' -delete


download_pascal_voc_2010_dataset:
	wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar" -O $(DATASET_TEMP_DIR)/pascal_voc_2010.tar
	tar -C $(DATASET_TEMP_DIR) -xf $(DATASET_TEMP_DIR)/pascal_voc_2010.tar
	rm $(DATASET_TEMP_DIR)/pascal_voc_2010.tar


remove_extra_images:
	poetry run $(PYTHON_EXEC) scripts/remove_extra_images_from_pascal.py


upload_data_to_clearml:
	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
	clearml-data add --files $(DATASET_TEMP_DIR)
	clearml-data close --verbose


run_training:
	poetry run $(PYTHON_EXEC) -m src.train
