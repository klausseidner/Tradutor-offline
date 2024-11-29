#####################################################################################################################
#
# Sistema de tradução de voz em tempo real
# Descrição: Este script realiza a tradução de voz em tempo real utilizando o modelo de tradução MarianMT.
#
# Desenvolvido por: Klaus Seidner
#
#####################################################################################################################

#####################################################################################################################
# Importações de bibliotecas necessárias
#####################################################################################################################
from Install_req import install_missing_packages # Importamos a função de verificação e instalação
# Executamos a função de verificação e instalação dos pacotes
install_missing_packages() 

# Importamos as bibliotecas instaladas
import os # Para lidar com o sistema operacional
import torch # Para trabalhar com tensores
import pyaudio # Para capturar o áudio
import speech_recognition as sr # Para reconhecer o áudio
from transformers import MarianMTModel, MarianTokenizer, AutoConfig # Para carregar o modelo de tradução
from TTS.api import TTS # Para gerar o áudio da tradução
import wave as wavfile # Para salvar o áudio da tradução

#####################################################################################################################
# Configuração do modelo de tradução
#####################################################################################################################
# Configuração de GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Verificamos se há GPU disponível
print(f"Using device: {DEVICE}") # Mostramos o dispositivo usado

# Lista de idiomas suportados pelo MarianMT
LANGUAGE_PAIRS = [
    "en-de", # Inglês para Alemão
    "de-en", # Alemão para Inglês
    "pt-de", # Português para Alemão
    "de-pt", # Alemão para Português
    "en-pt", # Inglês para Português
    "pt-en", # Português para Inglês
    "fr-en", # Francês para Inglês
    "en-fr", # Francês para Inglês
    "es-en", # Espanhol para Inglês
    "en-es", # Espanhol para Inglês
    "pt-fr", # Português para Francês
    "fr-pt" # Francês para Português
]

# Mapeamento de idiomas para nomes
LANGUAGE_MAP = {
    "en": "English",
    "de": "German",
    "pt": "Portuguese",
    "fr": "French",
    "es": "Spanish"
}

#####################################################################################################################
# Função para exibir as opções de idiomas disponíveis para a tradução
#####################################################################################################################
def display_language_options():
    print("\nAvailable languages:") # Mostramos os idiomas disponíveis
    for lang_code, lang_name in LANGUAGE_MAP.items(): # Iteramos sobre os pares de idiomas
        print(f"{lang_code}: {lang_name}") # Mostramos os nomes dos idiomas

#####################################################################################################################
# Função para escolher os idiomas
#####################################################################################################################
def get_language_selection():
    display_language_options() # Mostramos as opções de idiomas
    src_lang = input("Enter the code for the first language (e.g. 'pt' for Portuguese): ").strip() # Recebe o código do primeiro idioma
    tgt_lang = input("Enter the code for the second language (e.g. 'de' for German):").strip() # Recebe o código do segundo idioma
    pair = f"{src_lang}-{tgt_lang}" # Montamos o par de idiomas
    if pair not in LANGUAGE_PAIRS: # Verificamos se o par de idiomas é suportado
        print("Language pair not supported! Please try again.") # Mostramos uma mensagem de erro
        return get_language_selection() # Chamamos a função novamente para escolher os idiomas
    return src_lang, tgt_lang # Retornamos os códigos dos idiomas

#####################################################################################################################
# Função para carregar o modelo de tradução
#####################################################################################################################
def load_translation_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}" # Montamos o nome do modelo de tradução
    try: # Tentamos carregar o modelo
        AutoConfig.from_pretrained(model_name) # Carregamos a configuração do modelo
    except Exception: # Se houve um erro ao carregar o modelo
        print(f"Downloading translation template '{model_name}'...") # Baixamos o modelo
    tokenizer = MarianTokenizer.from_pretrained(model_name) # Carregamos o tokenizer do modelo
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE) # Carregamos o modelo de tradução
    return model, tokenizer # Retornamos o modelo e o tokenizer do modelo

#####################################################################################################################
# Função para carregar o modelo de síntese de fala
#####################################################################################################################
def load_tts_model(language_code):
    tts_models = {
        "pt": "tts_models/pt-cv/vits",
        "de": "tts_models/de/thorsten/vits",
        "en": "tts_models/en/ljspeech/tacotron2-DDC",
        "fr": "tts_models/fr/mai/vits",
        "es": "tts_models/es/mai/vits"
    }
    if language_code not in tts_models: # Verificamos se a síntese de fala não é suportada
        raise ValueError(f"Speech synthesis not supported for language '{language_code}'.") # Lançamos uma exceção para informar o erro
    return TTS(model_name=tts_models[language_code], progress_bar=False, gpu=torch.cuda.is_available()) # Retornamos o modelo de síntese de fala

#####################################################################################################################
# Função para capturar a fala do dispositivo
#####################################################################################################################
def recognize_speech_from_device(device_index, language):
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        print(f"[{LANGUAGE_MAP[language].upper()}] Waiting for speech...")
        try:
            audio = recognizer.listen(source)
            print(f"[{LANGUAGE_MAP[language].upper()}] Processing...")
            text = recognizer.recognize_google(audio, language=language)
            print(f"[{LANGUAGE_MAP[language].upper()}] Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print(f"[{LANGUAGE_MAP[language].upper()}] It was not possible to understand the speech.")
        except sr.RequestError:
            print(f"[{LANGUAGE_MAP[language].upper()}] Error accessing speech recognition service.")
    return None

#####################################################################################################################
# Função para traduzir o texto
#####################################################################################################################
def translate_text(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE) # Tokenizamos o texto
    translated = model.generate(**tokens) # Traduzimos o texto
    return tokenizer.decode(translated[0], skip_special_tokens=True) # Decodificamos a tradução

#####################################################################################################################
# Função para sintetizar o áudio
#####################################################################################################################
def play_audio_on_device(wav_file, device_index):
    audio = pyaudio.PyAudio() # Criamos um objeto PyAudio
    wf = wave.open(wav_file, 'rb') # Abrindo o arquivo de áudio
    # Configurando o stream de áudio para escrever no dispositivo de saída
    stream = audio.open( # Abrindo o stream de áudio
        format=audio.get_format_from_width(wf.getsampwidth()), # Obtendo o formato do áudio
        channels=wf.getnchannels(), # Obtendo o número de canais
        rate=wf.getframerate(), # Obtendo a taxa de amostragem
        output=True, # Definindo que o stream de áudio é de saída
        output_device_index=device_index # Definindo o dispositivo de saída
    )
    data = wf.readframes(1024) # Lendo os frames do arquivo de áudio
    while data: # Loop enquanto há dados no arquivo de áudio
        stream.write(data) # Escrevendo os frames no stream de áudio
        data = wf.readframes(1024) # Lendo os próximos frames do arquivo de áudio
    stream.stop_stream() # Parando o stream de áudio
    stream.close() # Fechando o stream de áudio
    audio.terminate() # Terminando o PyAudio

#####################################################################################################################
# Função para sintetizar o áudio e reproduzir
#####################################################################################################################
def synthesize_and_play(text, tts_model, language, device_index):
    print(f"[{LANGUAGE_MAP[language].upper()}] Generating audio...") # Gerando o áudio
    output_file = f"output_{language}.wav" # Definindo o nome do arquivo de saída do áudio
    tts_model.tts_to_file(text=text, file_path=output_file) # Gerando o áudio
    play_audio_on_device(output_file, device_index) # Reproduzindo o áudio

#####################################################################################################################
# Função principal do sistema
#####################################################################################################################
def main():
    src_lang, tgt_lang = get_language_selection() # Escolhendo os idiomas
    src_to_tgt_model, src_to_tgt_tokenizer = load_translation_model(src_lang, tgt_lang) # Carregando o modelo 1 de tradução
    tgt_to_src_model, tgt_to_src_tokenizer = load_translation_model(tgt_lang, src_lang) # Carregando o modelo 2 de tradução
    tts_src = load_tts_model(src_lang) # Carregando o modelo de síntese de fala do idioma do Usuário 1
    tts_tgt = load_tts_model(tgt_lang) # Carregando o modelo de síntese de fala do idioma do Usuário 2
    audio = pyaudio.PyAudio() # Criando um objeto PyAudio
    print("Available audio devices:") # Listando os dispositivos de áudio disponíveis
    for i in range(audio.get_device_count()): # Loop para listar os dispositivos de áudio
        device_info = audio.get_device_info_by_index(i) # Obtendo informações do dispositivo de áudio
        print(f"ID {i}: {device_info['name']}") # Mostrando os IDs e nomes dos dispositivos de áudio
    user1_mic_index = int(input("Enter User 1's microphone ID: ")) # Obtendo o ID do microfone do Usuário 1
    user2_mic_index = int(input("Enter User 2's microphone ID: ")) # Obtendo o ID do microfone do Usuário 2
    user1_spk_index = int(input("Enter User 1's output device ID: ")) # Obtendo o ID do dispositivo de saída do Usuário 1
    user2_spk_index = int(input("Enter User 2's output device ID:")) # Obtendo o ID do dispositivo de saída do Usuário 2
    while True: # Loop infinito para continuar a conversação
        user1_text = recognize_speech_from_device(user1_mic_index, src_lang) # Capturando a fala do Usuário 1
        if user1_text: # Se a fala do Usuário 1 foi capturada
            translated_to_user2 = translate_text(user1_text, src_to_tgt_model, src_to_tgt_tokenizer) # Traduzindo a fala do Usuário 1 para o idioma do Usuário 2
            synthesize_and_play(translated_to_user2, tts_tgt, tgt_lang, user2_spk_index) # Sintetizando e reproduzindo a fala do Usuário 2 com o áudio do Usuário 1
        user2_text = recognize_speech_from_device(user2_mic_index, tgt_lang) # Capturando a fala do Usuário 2
        if user2_text: # Se a fala do Usuário 2 foi capturada
            translated_to_user1 = translate_text(user2_text, tgt_to_src_model, tgt_to_src_tokenizer) # Traduzindo a fala do Usuário 2 para o idioma do Usuário 1
            synthesize_and_play(translated_to_user1, tts_src, src_lang, user1_spk_index) # Sintetizando e reproduzindo a fala do Usuário 1 com o áudio do Usuário 2

#####################################################################################################################
# Executando a função principal do sistema
#####################################################################################################################
if __name__ == "__main__": # Executando o script como o principal script
    main() # Executando a função principal do sistema

#####################################################################################################################