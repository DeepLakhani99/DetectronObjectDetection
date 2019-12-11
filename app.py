from flask import Flask, request, render_template
app = Flask(__name__)
import cv2
import numpy as np
import base64



from image_recognition import get_output
@app.route('/', methods=['GET', 'POST'])
def hello_world():
	if request.method == 'GET':
		return render_template('index.html', value='hi')
	if request.method == 'POST':
		print(request.files)
		if 'file' not in request.files:
			print('file not uploaded')
			return
		file = request.files['file']
		image = file.read()

		output=get_output(image)
		cv2.imshow('output',output)
		cv2.waitKey(0)
				

		return render_template('result.html',data=output, mimetype='image/jpg')
		'''
		file = base64.b64decode()
		nparr = npfromstring(file, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
		output=get_output(img)
		output=cv2.imdecode(output, cv2.IMREAD_ANYCOLOR)
		'''
		'''final_image=get_output(image_bytes=image)
		return render_template('result.html', flower=output, category=category)
'''
'''
		image = file.read()
		category, flower_name = get_flower_name(image_bytes=image)
		get_flower_name(image_bytes=image)
		tensor = get_tensor(image_bytes=image)
		print(get_tensor(image_bytes=image))
		return render_template('result.html', flower=flower_name, category=category)'''

if __name__ == '__main__':
	app.run(debug=True)