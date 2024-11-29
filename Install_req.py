#####################################################################################################################
# Instalador de bibliotecas
# Desenvolvido por: Klaus Seidner
#####################################################################################################################
import subprocess # Para executar comandos externos
import sys # Para lidar com argumentos de entrada

# Lista de pacotes necessários
REQUIRED_PACKAGES = [
    "torch", "transformers", "pyaudio", "speechrecognition", "TTS", "simpleaudio"
]

# Função para verificar e instalar pacotes
def install_missing_packages():
    for package in REQUIRED_PACKAGES: # Iteramos sobre os pacotes necessários
        try: # Tentamos importar o pacote
            __import__(package) # Se não há erro, o pacote está instalado
        except ImportError: # Se há um erro, o pacote não está instalado
            print(f"Package '{package}' not found. Instaling...") # Atualizamos a lista de pacotes necessários
            subprocess.check_call([sys.executable, "-m", "pip", "install", package]) # Executamos o comando de instalação

#####################################################################################################################