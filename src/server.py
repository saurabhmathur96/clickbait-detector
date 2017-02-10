from flask import Flask, jsonify, request
from detect import predictor



app = Flask(__name__)

@app.route("/detect", methods=["GET"])
def detect ():
    headline = request.args.get("headline", "")
    clickbaitiness = predictor.predict(headline)
    return jsonify({ "clickbaitiness": round(clickbaitiness * 100, 2) })


if __name__ == "__main__":
    app.run()
