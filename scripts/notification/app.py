from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'xyzg@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'xyz'         # Replace with your 16 character email's app password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

@app.route('/send_mail', methods=['POST'])
def send_mail():
    try:
        data = request.get_json()
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')

        if not email or not subject or not message:
            return jsonify({'error': 'Missing required fields'}), 400

        msg = Message(subject=subject, sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = message
        mail.send(msg)

        return jsonify({'message': 'Email sent successfully!'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
