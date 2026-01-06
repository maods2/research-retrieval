.PHONY: train app lint lint-diff clean retrival-all-models retrival-all-datasets-models

# Default configuration
CONFIG ?= default_train_config.yaml

# ============================
# Training Targets
# ============================
train:
	python3 src/main.py --config configs/$(CONFIG) --pipeline train

# ============================
# Evaluation Targets
# ============================
eval:
	python3 src/main.py --config configs/$(CONFIG) --pipeline eval

# ============================
# Training All Models on All Datasets	
# ============================

# datasets="skin-cancer glomerulo bracs ovarian-cancer lung-colon crc-val-he-7k";
#models="uni_fsl uni2h_fsl phikon_fsl phikon_v2_fsl virchow_fsl virchow_v2_fsl";
train-all-models:
	datasets="glomerulo"; \
	models="uni2h_fsl phikon_fsl phikon_v2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config test_configs/$$dataset/$$model.yml --pipeline train; \
		done; \
	done

# models="uni_fsl uni2h_fsl phikon_fsl phikon_v2_fsl virchow_fsl virchow_v2_fsl";
train-metric-all-models:
	datasets="glomerulo"; \
	models="uni_fsl uni2h_fsl phikon_fsl phikon_v2_fsl virchow_fsl virchow_v2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config test_configs/lwe/$$dataset/$$model.yml --pipeline train; \
		done; \
	done
# ============================
# Download Datasets
# ============================
# #gdown https://drive.google.com/uc?id=14GaaCw7og5jqwsBggb52EgVQKBwOJQpV &&
download-datasets:
	cd ./datasets && \
	gdown https://drive.google.com/drive/folders/1OV2dAb76InpUZoVa-HWRS_dzXGH9qv0m && \
	unzip bracs.zip && \
	unzip crc-val-he-7k.zip && \
	unzip lung-colon.zip && \
	unzip ovarian-cancer-subtypes.zip && \
	unzip skin-cancer.zip && \
	unzip ubc-ovarian-cancer.zip


# ============================
# Download Models
# ============================
download-virchow2:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1kBwDBUA85wo7IQS54WFcBxHnKd2cHbOg && \
	unzip Virchow2.zip && \
	rm -rf Virchow2.zip
download-uni2-h:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1FisQEXGLm5e0gWE2o0-jxWuef757GtDJ && \
	unzip UNI2-h.zip && \
	rm -rf UNI2-h.zip
download-uni:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1NV4dKyaOVmMtr-P_YP_p58KTLZqX_GZJ && \
	unzip uni.zip && \
	rm -rf uni.zip
download-phikon-v2:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1ip3sTjGoMWpGfcheNpaizhbLHQqUpDtM && \
	unzip phikon-v2.zip && \
	rm -rf phikon-v2.zip
download-phikon:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=12jdPlh2gDTVZflMc8SEzPos1XRDTains && \
	unzip phikon.zip && \
	rm -rf phikon.zip

download-models:
	# make download-virchow2
	make download-uni2-h
	make download-uni
	make download-phikon-v2
	make download-phikon


# ============================
# Linting Targets
# ============================
lint:
	blue ./src && isort ./src

lint-diff:
	blue --check --diff ./src && isort --check --diff ./src

# ============================
# Cleaning Target
# ============================
clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;


