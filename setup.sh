if [ ! -d "env" ]; then
	python3.8 -m venv "env"
	source env/bin/activate
	pip install -r requirements.txt
	pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
	pre-commit install
	huggingface-cli login
	wandb login
else 
	source env/bin/activate
fi
source .env
export PATH="$(pwd)/lib/bin:$PATH"


