.PHONY: setup corpus tokenizer data train chat-data chat-train serve test sample clean

setup:            ## install python dependencies
	pip install -r requirements.txt

corpus:           ## clone sources + build the Devanagari corpus
	git clone --depth 1 https://github.com/tokushige-koyasan/gretil-corpus data/raw/gretil || true
	git clone --depth 1 https://github.com/cltk/sanskrit_text_dcs data/raw/dcs || true
	python scripts/build_corpus.py --max-chars 6000000 --out data/clean

tokenizer:        ## train the akṣara-BPE tokenizer
	python scripts/train_tokenizer.py --vocab-size 4096 --sample 2500000

data:             ## encode the corpus into token arrays
	python scripts/prepare_data.py --data data/clean --out data/bin

train:            ## pretrain the LM (~45 min on 2 CPU cores)
	python scripts/train_model.py --steps 1900 --batch 16 --ctx 256

chat-data:        ## build chat-format pairs
	python scripts/build_chat_data.py
	python scripts/prepare_data.py --data data/chat --out data/bin_chat

chat-train:       ## adapt the model to the <user>/<assistant> template
	python scripts/train_model.py --resume artifacts/model.pt --bin-dir data/bin_chat \
		--steps 300 --lr 4e-4 --warmup 20 --reset-steps

serve:            ## run the chat interface
	python api/main.py

test:             ## run the test suite
	python tests/test_sanskrit.py

sample:           ## sample from the trained model
	python scripts/sample.py "धर्मक्षेत्रे कुरुक्षेत्रे" --n 3 --mode continuation

clean:            ## remove generated data and checkpoints
	rm -rf data/raw data/bin data/bin_chat data/chat data/*.txt artifacts/*.pt
