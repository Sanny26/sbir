import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
p_parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, p_parentdir)

from django.shortcuts import render, reverse, redirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .forms import ImSearchForm

from settings import docs_folder
from settings import codebook_size
from settings import tfidf_model
from settings import sc_model, hog_model, spark_model
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from bof import get_words, get_models
from tfidf import find_matching_docs, find_images
import pickle

import base64
import pdb
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import urllib.parse


doc_names, matrix = pickle.load(open(tfidf_model, "rb"))
models = get_models([hog_model, sc_model, spark_model])


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
    img = np.array(img)
    if len(img.shape) == 3:
        img = rgb2gray(img)
    thresh = threshold_otsu(img)
    img = (img > thresh).astype(int)
    print("Processing image")
    words = get_words(img, models, [codebook_size, codebook_size, codebook_size], True)
    words = list(set(words))
    docs_match = find_matching_docs(matrix, words, doc_names)
    images = find_images(docs_match)
    filepaths = ["static/images/" + x for x in images]
    return filepaths


def results(request):
    template_name = 'main/results.html'
    context = {}
    img = request.session['image']  # rgb image
    results = search(img)
    context['qimg'] = reverse('show_image')
    context['results1'] = results[0:5]
    context['results2'] = results[5:10]
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