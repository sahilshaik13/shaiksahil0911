from flask import Flask, render_template_string

app = Flask(__name__, template_folder='.')

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Stylobot Flask App</title>
    <link rel="stylesheet" href="https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/themes/df-messenger-default.css">
    <script src="https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/df-messenger.js"></script>
    <style>
        df-messenger {
            z-index: 999;
            position: fixed;
            --df-messenger-font-color: #000;
            --df-messenger-font-family: Google Sans;
            --df-messenger-chat-background: #f3f6fc;
            --df-messenger-message-user-background: #d3e3fd;
            --df-messenger-message-bot-background: #fff;
            bottom: 16px;
            right: 16px;
        }
    </style>
</head>
<body>
    <h1>Welcome to the StyloBot Flask App!</h1>
    <df-messenger
        location="us-central1"
        project-id="plexiform-plane-462910-p9"
        agent-id="9bc44d89-0b59-4257-8d7e-0df7b2d29500"
        language-code="en"
        max-query-length="-1">
        <df-messenger-chat-bubble
            chat-title="Stylobot">
        </df-messenger-chat-bubble>
    </df-messenger>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html_content)

if __name__ == "__main__":
    app.run(debug=True)