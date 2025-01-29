import nbformat
from flask import Flask, jsonify, render_template
from pathlib import Path
import threading
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from nbconvert.preprocessors import ExecutePreprocessor
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__, template_folder='new templates', static_folder='new templates/static')

def fetch_data():
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    # Rename columns if necessary
    df = df.rename(columns={
        'YourNewIdentityColumn': 'Identity',
        'YourNewEmotionColumn': 'Emotion'
    })
    return df


#Set up Google Sheets credentials
scope = ['https://www.googleapis.com/auth/drive']
json_keyfile_path = r'../face-recognition-new-credentials.json'  # Update this with the path to your JSON keyfile
creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_path, scope)
client = gspread.authorize(creds)

# Specify the Google Sheets document by its URL
sheet_url = "https://docs.google.com/spreadsheets/d/1Iqfx7UThkXlXW2fAvXWyEOejkNIQTQhsOaGqphVdCJ8/edit?gid=0#gid=0"

# Open the Google Sheets document
doc = client.open_by_url(sheet_url)
sheet = doc.worksheet("Face")



@app.route('/face-recognition-data')
def index():
    df = fetch_data()

    # Update column names if they're different in your new sheet
    face_counts = df['Identity'].value_counts().reset_index()
    face_counts.columns = ['Identity', 'Count']
    face_fig = px.bar(face_counts, x='Identity', y='Count', title='Your New Face Detection Title', color='Identity',
                      color_discrete_sequence=px.colors.qualitative.Bold)

    mood_counts = df['Emotion'].value_counts().reset_index()
    mood_counts.columns = ['Emotion', 'Count']
    mood_fig = px.bar(mood_counts, x='Emotion', y='Count', title='Your New Mood Detection Title', color='Emotion',
                      color_discrete_sequence=px.colors.qualitative.Pastel)

    # Convert plots to HTML
    face_plot = pio.to_html(face_fig, full_html=False)
    mood_plot = pio.to_html(mood_fig, full_html=False)

    return render_template('index.html', face_plot=face_plot, mood_plot=mood_plot)




@app.route('/face-recognition')
def face_recognition():
    return render_template('facial_recognition.html')



@app.route('/run-face-recognition', methods=['GET'])
def run_face_recognition():
    notebook_path = "../Face Recognition Model Final Version.ipynb"
    if not Path(notebook_path).is_file():
        return jsonify({"error": "Notebook not found"}), 404

    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        # Set up the ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Execute the notebook
        threading.Thread(ep.preprocess(notebook, {'metadata': {'path': Path(notebook_path).parent}})).start()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)