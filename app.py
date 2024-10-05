import os

import flask
import google.generativeai as genai
from dotenv import load_dotenv

app = flask.Flask(__name__)


load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash-latest")


@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"


@app.get("/test")
@app.get("/test/<name>")
def test(name=None):
    args = flask.request.args
    resp_json = {"args": args, "name": name}
    return resp_json, 200


@app.get("/genai/test")
def genai_test():
    query = flask.request.args["query"]
    if not query:
        return {"error": "query is required"}, 400
    response = model.generate_content(query)
    return response.text, 200


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
