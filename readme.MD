# 🎤 Tradutor em tempo real (Offline)

<p align="center">
  <img src="https://github.com/klausseidner/Tradutor-offline/blob/main/logo.png" width="20%" alt="Logo do Projeto">
</p>


## 📖 Descrição

Este sistema realiza a **tradução de voz em tempo real**, utilizando o modelo de tradução **MarianMT** da biblioteca `transformers` e a síntese de fala por meio da biblioteca `TTS`. Ele permite que duas pessoas se comuniquem em diferentes idiomas, com a tradução sendo feita instantaneamente entre o áudio capturado e reproduzido.

<p align="center">
	<img src="https://img.shields.io/github/license/klausseidner/Tradutor-offline?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/klausseidner/Tradutor-offline?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/klausseidner/Tradutor-offline?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/klausseidner/Tradutor-offline?style=flat&color=0080ff" alt="repo-language-count">
</p>


## 🚀 Funcionalidades

- 🎙️ Captura de áudio de dois usuários em tempo real.
- 🌍 Tradução de voz em tempo real entre diferentes pares de idiomas (e.g., Português para Alemão, Inglês para Espanhol).
- 🔊 Reprodução do áudio traduzido com síntese de fala.
- 📶 Não necessita de conexão com a internet após a primeira execução.


## 📦 Requisitos

Os seguintes pacotes Python são instalados automaticamente:

- `torch`
- `transformers`
- `pyaudio`
- `speechrecognition`
- `TTS`
- `simpleaudio`


## 🛠️ Como Usar

1. Clone ou baixe este repositório.
2. Conecte dois microfones e dois dispositivos de saída de áudio (alto-falantes ou fones de ouvido).
3. Execute o script Python:

```bash
   python3 ./main.py
```

- O sistema irá solicitar que você selecione os idiomas para a tradução.
- Após isso, o sistema começará a capturar o áudio dos dois usuários, traduzindo e reproduzindo o áudio traduzido para o outro usuário.


## 💡 Exemplo de Execução
Escolha dois idiomas:

- Idioma 1: Português
- Idioma 2: Alemão

Fluxo de Tradução:

- O Usuário 1 fala em Português.
- O sistema traduz para Alemão e reproduz o áudio para o Usuário 2.
- O Usuário 2 fala em Alemão.
- O sistema traduz para Português e reproduz o áudio para o Usuário 1.

**O sistema continuará a tradução em tempo real até que você o interrompa.**


## 📂 Estrutura do Código

```sh
├── install_req.py - Script para instalar as dependências necessárias.
└── main.py - Script principal que executa a captura, tradução e reprodução do áudio.
```


## 👤 Autor

- [@klausseidner](https://www.github.com/klausseidner) (Klaus Seidner)


## 🤝 Contribuições

Contribuições são bem-vindas! Se você encontrar problemas ou tiver melhorias para sugerir, sinta-se à vontade para abrir um issue ou um pull request.


## 🍺 Me Pague uma cerveja! 🍺

[![BTC Wallet](http://img.shields.io/badge/Bitcoin-000000?style=flat&logo=bitcoin&logoColor=white)](https://www.blockchain.com/btc/address/bc1qtfh4asd7jhyxxpnk0254c2tr6zy4p3aagr9lnc)
[![ETH Wallet](http://img.shields.io/badge/Ethereum-000000?style=flat&logo=ethereum&logoColor=white)](https://www.blockchain.com/eth/address/0x4bdebD8AA648a88f06Acc7944Ab852676eF059d1)
[![SOL Wallet](http://img.shields.io/badge/Solana-000000?style=flat&logo=solana&logoColor=white)](https://solscan.io/account/2NWJyYUx4YgdAWkr4pbjbqQvtSGCFH44mqJ1gHnfxu3L)


## 📜 Licença

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
