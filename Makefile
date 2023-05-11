SHELL := /bin/zsh
AUTOSUGGESTIONS_DIR := /home/vscode/.oh-my-zsh/custom/plugins/zsh-autosuggestions

all:
	make add-path
	make add-fonts
	make add-star
	make add-autosuggestions
	make add-zoxide

update:
	sudo apt update

add-path:
	echo 'export PYENV_ROOT="${HOME}/.pyenv"' >> ~/.zshrc
	echo 'export PATH="$${PYENV_ROOT}/shims:$${PYENV_ROOT}/bin:$${HOME}/.local/bin:$$PATH"' >> ~/.zshrc

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
	# poetry run flake8 dev/scripts
	# poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports dev/scripts

format:
	poetry run isort -v src
	poetry run black src
	# poetry run isort -v dev/scripts
	# poetry run black dev/scripts