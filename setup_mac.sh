if [ ! -d "env" ]; then
	python3 -m venv "env"
	source env/bin/activate
	pip3 install -r requirements.txt
	pip3 install torch torchvision torchaudio
	pre-commit install
	huggingface-cli login
	wandb login
fi
source env/bin/activate
source .env
