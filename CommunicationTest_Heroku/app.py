import json
import datetime
import numpy as np

import phe

from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/compute_time/', methods=['POST'])
def compute_time():
	json_data = request.get_json()
	x_enc = json.loads(json_data)

	x = np.asarray([phe.EncryptedNumber.deserialize(x_enc[i]) for i in range(len(x_enc))])

	time_post = datetime.datetime.utcnow().isoformat()
	result = {'time_post': time_post}
	response = app.response_class(
		response=json.dumps(result),
		status=200,
		mimetype='application/json'
	)
	return response


if __name__ == '__main__':
	app.run(debug=True)
