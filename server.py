from flask import Flask, render_template, request
from VideoDetector.video_detection import VideoDetection
app = Flask("PerformerPulse")
@app.route('/videoDetector')
def video_detector():
    url_to_analyze = request.args.get('urlToAnalyze')
    result = VideoDetection(url_to_analyze).generate_response()
    return result
@app.route("/")
def render_index_page():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
