from flask import *
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import subprocess
import threading
import time
import csv
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth
import scapy.all as scapy
import os
from collections import deque
from queue import Queue
import glob
from flask_cors import CORS
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import pandas as pd

#for training model
from model import train_ml_model



from wifi_signal import ap as wifi_signal_blueprint


packet_queue = Queue()
cur_ip = None

class CommandThread(threading.Thread):
    def __init__(self, interface):
        super(CommandThread, self).__init__()
        self.interface = interface
        self._stop_event = threading.Event()

    def run(self):
        commands = ["g++ capture.c -lpcap -lcjson -lcurl -o cap.out", f"sudo ./cap.out {self.interface}"]
        for command in commands:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if self._stop_event.is_set():
                break

command_thread = None

app = Flask(__name__)
app.secret_key = 'my_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
socketio = SocketIO(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
CORS(app)

#for train model
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    if user:
        return User(id=user[0], username=user[1], password=user[2])
    return None

conn = sqlite3.connect('users.db')
c = conn.cursor()
firewall_ip = c.execute('SELECT IP FROM ips').fetchone()
conn.close()

OPNSENSE_HOST = f"http://{firewall_ip[0]}"
API_KEY = "4DxnKG0eBH8YRGqVhNIe8cM2UjqvdSOvXq65nyvY5gkM0ZOoAaOOl3kTPXmrI1KEB6qQL3nfkN8a2+s8"
API_SECRET = "xevRWAx/eH6YN+XIu01ycGSY02ybWgcmi6wBRZ/5viZFW5OhJMECEsca/6u65ABYFsHlZ25SyOoIXN2J"
auth = (API_KEY, API_SECRET)

interfaces = []  # Initialize interfaces before usage
sent_bytes = []

url = f"{OPNSENSE_HOST}/api/diagnostics/interface/getInterfaceStatistics"

@app.route("/")
@login_required
def home():
    return render_template("index.html",firewall_ip=firewall_ip[0])

@app.route("/get-interfaces")
def get_interfaces():
    global sent_bytes, interfaces
    interfaces = []
    packets_ps = requests.get(url, auth=(API_KEY, API_SECRET))
    data = packets_ps.json()
    for interface in data['statistics']:
        if 'Loopback' not in interface and ':' not in interface:
            interfaces.append(interface)
    sent_bytes = [[] for i in range(len(interfaces))]
    return jsonify({'interfaces': interfaces})

@app.route('/firewalltraffic', methods=['GET'])
def get_traffic_value():
    try:
        response = requests.get(url, auth=(API_KEY, API_SECRET))
        data = response.json()
        stats = data['statistics']
        traffic_data = {}
        c = 0
        for interface in stats:
            if 'Loopback' not in interface and ':' not in interface:
                sent_bytes[c].append(stats[interface]['sent-bytes'])
                c += 1
        if len(sent_bytes[0]) >= 2:
            for i in range(len(sent_bytes)):
                l = len(sent_bytes[i])
                traffic_data[interfaces[i]] = abs(sent_bytes[i][l-1] - sent_bytes[i][l-2])
        
        return jsonify(traffic_data)
    except Exception as e:
        return jsonify(f"Error: {e}")
    
@app.route('/download/<filename>/<format>')
@login_required
def download_file(filename, format):
    csv_path = os.path.join('static', 'csv', filename)
    
    if format == 'csv':
        return send_file(csv_path, as_attachment=True)
    elif format == 'xlsx':
        xlsx_path = csv_path.replace('.csv', '.xlsx')
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            df.to_excel(xlsx_path, index=False)
            return send_file(xlsx_path, as_attachment=True)
        except Exception as e:
            return jsonify({'error': f'Failed to convert CSV to XLSX: {e}'}), 400
    elif format == 'log':
        log_path = csv_path.replace('.csv', '.log')
        with open(csv_path, 'r') as csv_file:
            with open(log_path, 'w') as log_file:
                log_file.write(csv_file.read())
        return send_file(log_path, as_attachment=True)
    elif format == 'json':
        json_path = csv_path.replace('.csv', '.json')
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            df.to_json(json_path, orient='records', lines=True)
            return send_file(json_path, as_attachment=True)
        except Exception as e:
            return jsonify({'error': f'Failed to convert CSV to JSON: {e}'}), 400
    else:
        return jsonify({'error': 'Invalid format requested'}), 400


@app.route('/addrule', methods=['GET', 'POST'])
@login_required
def add_rule():
    return render_template('add_rule.html')

@app.route("/firewallip", methods=['GET', 'POST'])
@login_required
def edit_firewall_ip():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    if request.method == "POST":
        ip = request.form["ip"]
        c.execute('INSERT INTO ips (IP) VALUES (?)', (ip,))
        conn.commit()
    cur_ip = c.execute("SELECT * FROM ips").fetchall()
    print(cur_ip)
    conn.close()
    return render_template("editfirewallip.html", cur_ip=cur_ip)

@app.route("/update_firewall_ip/<int:id>", methods=['GET', 'POST'])
def update_firewall_ip(id):
    id = int(id)
    global OPNSENSE_HOST
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    firewall_ip = c.execute('SELECT IP FROM ips').fetchone()
    if request.method == "POST":
        ip = request.form["ip"]
        c.execute('UPDATE ips SET IP = ? WHERE ID = ?', (ip, id))
        conn.commit()
        conn.close()
        OPNSENSE_HOST = f"http://{ip}"
        return redirect(url_for('edit_firewall_ip'))
    return render_template("update_firewall_ip.html", firewall_ip=firewall_ip[0])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            user_obj = User(id=user[0], username=user[1], password=user[2])
            login_user(user_obj)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/adduser", methods=["GET", "POST"])
@login_required
def adduser():
    return render_template("adduser.html")

@app.route("/viewlivepackets")
@login_required
def viewlivepackets():
    interfaces = scapy.get_if_list()
    print(interfaces)
    return render_template("interfaces.html", interfaces=interfaces)

@app.route("/packet", methods=["POST"])
def receive_packet():
    packet_json = request.get_json()
    packet_queue.put(packet_json)
    return "Packet received", 200

@app.route("/get_packets")
def get_packets():
    packets = []
    while not packet_queue.empty():
        packets.append(packet_queue.get())
    return jsonify(packets)

@app.route("/viewlivepackets/<string:interface>")
def viewliveinterface(interface):
    command_thread = CommandThread(interface)
    command_thread.start()
    return render_template("traffic.html", interface=interface)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback = request.form.get('feedback')  # Retrieve feedback from form data
        name = request.form.get('name')  # Assuming you have a way to get the username
        emoji = request.form.get('emoji')  # Retrieve the selected emoji
        
        print("Feedback Received:", feedback)  # Add logging to check if feedback is received

        # Use environment variables for sensitive information
        EMAIL_USER = os.getenv('EMAIL_USER', '125003239@sastra.ac.in')
        EMAIL_PASS = os.getenv('EMAIL_PASS', '09062003')

        if emoji == 'issue':  # Only send email if the emoji indicates an issue
            # Create a MIMEText object to represent the email
            msg = MIMEMultipart()
            me = EMAIL_USER

            try:
                you = "125003358@sastra.ac.in"
                subject = '🚩Feedback from {}'.format(name)
                msg['Subject'] = subject
                msg['From'] = me
                msg['To'] = you

                body = f"Feedback from {name}:\n{feedback}"
                msg.attach(MIMEText(body, 'plain'))

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(EMAIL_USER, EMAIL_PASS)
                    server.sendmail(me, [you], msg.as_string())

                print("Mail sent successfully")
                return jsonify({'success': True})  # Send success response
            except Exception as e:
                print(f"Failed to send email: {e}")
                return jsonify({'success': False, 'error': str(e)})  # Send failure response
        else:
            # Store the feedback in a file under static
            feedback_dir = os.path.join('static', 'feedback')
            os.makedirs(feedback_dir, exist_ok=True)
            feedback_file = os.path.join(feedback_dir, f"{name}_feedback.txt")
            with open(feedback_file, 'a') as f:
                f.write(f"{datetime.now()} - {emoji}: {feedback}\n")
            print("Feedback stored successfully")
            return jsonify({'success': True})  # Send success response

    else:
        return render_template("feedback.html")

#for logs
@app.route("/logs")
@login_required
def logs():
    csv_files = glob.glob(os.path.join('static\\csv', "*.csv"))
    csv_files = [os.path.basename(file) for file in csv_files]
    return render_template("logs.html", csv_files=csv_files)

#redirecting to model
@app.route('/model')
def model():
    return render_template("model.html")
#training model

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'message': 'Invalid file type'})

    model_type = request.form['model_type']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    try:
        if model_type == 'ml':
            trained_model_filename = train_ml_model(filename)
        elif model_type == 'dl':
            trained_model_filename = train_dl_model(filename)
        else:
            return jsonify({'success': False, 'message': 'Invalid model type'})

        return jsonify({'success': True, 'filename': trained_model_filename})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route("/download/<filename>")
def download_trained_model(filename):
    return send_file(filename, as_attachment=True)

# Register the wifi_signal blueprint
app.register_blueprint(wifi_signal_blueprint)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)