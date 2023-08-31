from flask import Flask, request, jsonify, render_template
import torch
import torchaudio
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        print("hello")
        s2st_target_language_codes = {
            "English" : "eng",
            "Modern Standard Arabic" : "arb",
            "Bengali" : "ben",
            "Catalan" : "cat",
            "Czech" : "ces",
            "Mandarin Chinese" : "cmn",
            "Welsh" : "cym",
            "Danish" : "dan",
            "German" : "deu",
            "Estonian" : "est",
            "Finnish" : "fin",
            "French" : "fra",
            "Hindi" : "hin",
            "Indonesian" : "ind",
            "Italian" : "ita",
            "Japanese" : "jpn",
            "Korean" : "kor",
            "Maltese" : "mlt",
            "Dutch" : "nld",
            "Western Persian" : "pes",
            "Polish" : "pol",
            "Portuguese" : "por",
            "Romanian" : "ron",
            "Russian" : "rus",
            "Slovak" : "slk",
            "Spanish" : "spa",
            "Swedish" : "swe",
            "Swahili" : "swh",
            "Telugu" : "tel",
            "Tagalog" : "tgl",
            "Thai" : "tha",
            "Turkish" : "tur",
            "Ukrainian" : "ukr",
            "Urdu" : "urd",
            "Northern Uzbek" : "uzn",
            "Vietnamese" : "vie",
        }

        target_language = s2st_target_language_codes[str(request.form['targetLanguage'])]

        return jsonify(validate_audio_file(request.files['audio'], target_language))

    except Exception as e:
        return jsonify({'error': str(e)})
    
def validate_audio_file(file_path, target_language):
    try:
        # Load the S2ST model
        s2st_model = torch.jit.load('unity_on_device.ptl')
        
        # Load the audio input from the request
        waveform, sample_rate = torchaudio.load(file_path)
        print("File is valid.")

        # Use the loaded model to perform speech-to-speech translation
        with torch.no_grad():
            text, units, waveform = s2st_model(waveform, tgt_lang=target_language)

        output_directory = 'static'
        os.makedirs(output_directory, exist_ok=True)

        # Save the output waveform to a file
        torchaudio.save(os.path.join(output_directory, 'output.wav'), waveform.unsqueeze(0), sample_rate=16000)

        # Prepare the response
        response = {'text': text, 'audio_url': 'static/output.wav'}
        return response
        
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    app.run(debug=True)
