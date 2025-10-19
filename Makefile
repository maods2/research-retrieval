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

train-all-models:
	datasets="skin-cancer glomerulo"; \
	models="phikon-v2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/$$model\_config.yaml --pipeline train; \
		done; \
	done

# ============================
# Download Datasets
# ============================
download-datasets:
	datasets="bracs crc-val-he-7k lung-colon skin-cancer ubc-ovarian-cancer"; \
	ids="1G6Db1DZ_t3dctc78XvQKLwpSR2VOFRt4 1ZlTFj3OGeSW-rx-wvYHROaRIZRgdsfCh 17NVQEcxBt5gCSllS4PUs6RSPaJr1VQyZ 1hs449HmRTXUb3xyDvtJOFR0IjA4gzIrz 1mLLWzdgKBRwL8wM2lPjSnIYgh2TO4YLB"; \
	i=0; \
	for dataset in $$datasets; do \
		id=$$(echo $$ids | cut -d' ' -f$$((i+1))); \
		make download-dataset DATASET=$$dataset ID=$$id; \
		i=$$((i+1)); \
	done

download-dataset:
	cd ./data && \
	gdown https://drive.google.com/uc?id=$(ID) && \
	unzip $(DATASET).zip && \
	rm -rf $(DATASET).zip


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


