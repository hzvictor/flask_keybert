from flask import Flask
from keybert import KeyBERT
from flask import request
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/words', methods=['POST'])
def post_words():  # put application's code here
    data = request.form.get('text')
    kw_model = KeyBERT(model='all-mpnet-base-v2')

    keyword = kw_model.extract_keywords(data,

                                        keyphrase_ngram_range=(0, 1),

                                        stop_words='english',

                                        highlight=False,

                                        top_n=2)
    print(data)
    return {
        "status": 0,
        "msg": "ok",
        "keyword": keyword
    }


@app.route('/test', methods=['POST'])
def test():  # put application's code here
    data = request.form.get('text')
    print(data)
    return {
        "status": 0,
        "msg": "ok",
        data: data,
        "keyword": [
            [
                "recommend hashtags",
                0.532
            ],
            [
                "hashtags step",
                0.4579
            ],
            [
                "hashtags",
                0.4131
            ]
        ],
    }


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True
    )
