SHELL := /bin/zsh
AUTOSUGGESTIONS_DIR := /home/vscode/.oh-my-zsh/custom/plugins/zsh-autosuggestions
POETRY_DIR := /home/vscode/.cache/pypoetry

.PHONY: help install update add-path add-fonts add-star add-autosuggestions add-zoxide lint format

all:
	make update
	make env
	make add-path
	make add-fonts
	make add-star
	make add-autosuggestions
	make add-zoxide

env:
	if [ ! -d "$(POETRY_DIR)" ]; then \
		poetry install; \
	fi

clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

update:
	sudo apt update

add-path:
	# echo 'export PYENV_ROOT="${HOME}/.pyenv"' >> ~/.zshrc
	# echo 'eval "$$(pyenv init -)"' >> ~/.zshrc \
	# echo 'export PATH="$${PYENV_ROOT}/shims:$${PYENV_ROOT}/bin:$${HOME}/.local/bin:$$PATH"' >> ~/.zshrc
	echo 'export PATH="$${HOME}/.local/bin:$$PATH"' >> ~/.zshrc

add-fonts:
	sudo apt update \
	&& sudo apt install -y --no-install-recommends \
	unzip \
	fontconfig \
	&& cd /usr/share/fonts \
	&& sudo wget https://github.com/ryanoasis/nerd-fonts/releases/download/v2.2.2/FiraCode.zip \
	&& sudo unzip -o FiraCode.zip \
	&& fc-cache -f -v \
	&& echo "fontconfig installed"

add-star:
	curl -ss https://starship.rs/install.sh | sh -s -- --yes \
	&& echo 'eval "$$(pyenv init -)"' >> ~/.zshrc \
	&& echo 'eval "$$(starship init zsh)"' >> ~/.zshrc \
	&& starship preset nerd-font-symbols -o ~/.config/starship.toml

add-autosuggestions: $(AUTOSUGGESTIONS_DIR)
	@if grep -q 'zsh-autosuggestions' ~/.zshrc; \
	then \
		echo "zsh-autosuggestions already present in plugins list"; \
	else \
		sed -i 's/plugins=(/plugins=(zsh-autosuggestions /' ~/.zshrc; \
		echo "zsh-autosuggestions added to plugins list"; \
	fi

$(AUTOSUGGESTIONS_DIR):
	sudo git clone https://github.com/zsh-users/zsh-autosuggestions $(AUTOSUGGESTIONS_DIR)

add-zoxide:
	curl -sS https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | bash \
	&& echo 'eval "$$(zoxide init zsh)"' >> ~/.zshrc

add-exa:
	sudo apt install exa \
	&& echo "alias lsd='exa -h --icons --long --sort=mod'" >> ~/.zshrc

lint:
	poetry run flake8 src
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports src
	poetry run flake8 src
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports src
	# poetry run flake8 dev/scripts
	# poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports dev/scripts

format:
	poetry run isort -v src
	poetry run black src
	# poetry run isort -v dev/scripts
	# poetry run black dev/scripts

# `show_logs` target: Run the MLflow server to visualize experiment logs
# Start the MLflow server with the specified configuration
# Set the URI for the backend store (where MLflow metadata is stored)
# Set the default root directory for storing artifacts (e.g., models, plots)
# Set the host for the MLflow server to bind to (localhost in this case)
show_logs:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 127.0.0.1
	
# `stop_server` target: Check if an MLflow server is running on port 5000 and shut it down if it is
# Find the process listening on port 5000, filter by 'mlflow', extract its process ID, and terminate it
stop_server:
	@-lsof -i :5000 -sTCP:LISTEN | grep 'mlflow' | awk '{ print $$2 }' | xargs -I {} kill {}

streamlit:
	streamlit run app.py

graphviz:
	sudo apt update && apt install -y graphviz