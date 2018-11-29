from django.shortcuts import render, reverse, redirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .forms import ImSearchForm

import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np 
import urllib.parse

# Create your views here.
def index(request):
	template_name = 'main/index.html'
	context = {}

	if request.method == 'POST':
		form = ImSearchForm(request.POST, request.FILES)
		if form.is_valid():
			fobj = request.FILES['query']
			jpeg_array = bytearray(fobj.read())
			img = cv2.imdecode(np.asarray(jpeg_array), 1)
			request.session['image'] = img.tolist()
			return redirect('results')
	else:
		form = ImSearchForm()
		context['form'] = form
	return render(request, template_name, context)

def search(img):
	result_paths = []
	return result_paths

def results(request):
	template_name = 'main/results.html'
	context = {}
	img = request.session['image'] ## rgb image
	results = search(img)
	context['qimg'] = reverse('show_image')
	context['results'] = results
	return render(request, template_name, context)

def show_image(request):
	qimg = np.array(request.session['image'])
	qimg = Image.fromarray(np.uint8(qimg))
	response = HttpResponse(content_type="image/png")
	qimg.save(response, "PNG")
	return response

def inputpad(request):
	template_name = 'main/input_pad.html'
	context = {}
	return render(request, template_name, context)

@csrf_exempt
def read_image(request):
	template_name = 'main/index.html'
	context = {}
	if request.is_ajax():
		dataURL = request.POST.get('dataURL')
		up = urllib.parse.urlparse(dataURL)
		head, data = up.path.split(',', 1)
		bits = head.split(';')
		mime_type = bits[0] if bits[0] else 'text/plain'
		charset, b64 = 'ASCII', False
		for bit in bits:
		    if bit.startswith('charset='):
		        charset = bit[8:]
		    elif bit == 'base64':
		        b64 = True
		imgData = base64.b64decode(	data)
		im = np.asarray(Image.open(BytesIO(imgData)))
		#cv2.imwrite( 'test.jpg', im)
		request.session['image'] = im.tolist()
		return redirect('results')
	return redirect('index')