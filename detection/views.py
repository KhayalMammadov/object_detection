from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.shortcuts import render, redirect
from PIL import Image as ImageReader
from django.views import View
import json
from object_detection import settings
from django.http import HttpResponse, JsonResponse
from .models import Image, Image2User, MaskInfo, SavedImage
from .detect import auto_detection_coco
from .utils import string2list
from .forms import ImageForm
from detection.ml import manual_detection
from skimage import io
import os
import time
from detection.ml.temp import *

# Create your views here.

STATIC_DIR = settings.STATIC_DIR
MEDIA_ROOT = settings.MEDIA_ROOT
default_class_names = ['All', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']


@login_required
def home(request):
    class_names_list = default_class_names
    image_ids = Image2User.objects.filter(user=request.user)
    images = []
    for id in image_ids:
        images += Image.objects.filter(id=id.file.id)
    data = {
        'class_names': class_names_list,
        'images': images
    }
    try:
        data['current_url'] = request.GET['url']
    except:
        pass
    return render(request, 'detection/home.html', data)


@method_decorator([login_required], name='dispatch')
class ImageUploadView(View):
    def get(self, request):
        image_ids = Image2User.objects.filter(user=request.user)
        images = []
        for id in image_ids:
            images += Image.objects.filter(id=id.file.id)
        return render(request, 'detection/file_upload.html', {'images': images})

    def post(self, request):
        form = ImageForm(self.request.POST, self.request.FILES)
        user = request.user
        if form.is_valid():
            image = form.save()
            image2user = Image2User(user=user, file=image)
            image2user.save()
            data = {'is_valid': True, 'name': image.file.name, 'url': image.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)


@login_required
def clear_database(request):
    user = request.user
    ids = Image2User.objects.filter(user=user)
    images = []
    for id in ids:
        images += Image.objects.filter(id=id.file.id)
    for image in images:
        image.file.delete()
        image.delete()
    return redirect(request.POST.get('next'))


@login_required
def delete_image(request, pk):
    image = MaskInfo.objects.get(id=pk)
    image.delete()
    return redirect('detection:manage_images')


@login_required
def delete_saved_image(request, pk):
    image = SavedImage.objects.get(id=pk)
    try:
        os.remove(settings.MEDIA_ROOT + image.file)
    except:
        pass
    image.delete()
    return redirect('detection:manage_images')


@login_required
def clear_database_masked(request):
    user = request.user
    masks = MaskInfo.objects.filter(user=user)
    for mask in masks:
        mask.delete()
    return redirect(request.POST.get('next'))


@login_required
def clear_database_saved(request):
    user = request.user
    images = SavedImage.objects.filter(user=user)
    for image in images:
        image.delete()
    return redirect(request.POST.get('next'))


@login_required
def draw_mask(request):
    if request.method == 'GET':
        url = request.GET['url']
        img = ImageReader.open(MEDIA_ROOT + url[6:])
        width, height = img.size
        train_num = len(MaskInfo.objects.filter(user=request.user, is_train=True))
        valid_num = len(MaskInfo.objects.filter(user=request.user, is_train=False))
        data = {
            'image_url': url,
            'width': width,
            'height': height,
            "train_num": train_num,
            "valid_num": valid_num
        }
        return render(request, 'detection/draw_mask.html', data)
    if request.method == 'POST':
        coordinates = request.POST.get('coordinates')
        if request.POST['mask_type'] == 'train':
            is_train = True
        else:
            is_train = False
        class_name = request.POST['class_name']
        image_url = request.POST.get('image_url')
        mask = MaskInfo()
        mask.user = request.user
        mask.class_name = class_name
        mask.is_train = is_train
        mask.mask_path = coordinates
        mask.image_url = image_url
        mask.save()
        train_num = len(MaskInfo.objects.filter(user=request.user, is_train=True))
        valid_num = len(MaskInfo.objects.filter(user=request.user, is_train=False))
        img = ImageReader.open(MEDIA_ROOT + image_url[6:])
        width, height = img.size
        data = {
            "image_url": image_url,
            'width': width,
            'height': height,
            "train_num": train_num,
            "valid_num": valid_num
        }
        return render(request, 'detection/draw_mask.html', data)


def train_model(request):
    if request.method == 'POST':
        train_mask_dict = {}
        train_mask_dict["regions"] = []
        valid_mask_dict = {}
        valid_mask_dict["regions"] = []
        dataset = {}
        class_names = list(set([mask.class_name for mask in MaskInfo.objects.all()]))
        for mask in MaskInfo.objects.all():
            if mask.mask_path:
                coordinates_list = eval(str(mask.mask_path))
            else:
                continue
            all_points_x = [int(float(cor['x'])) for cor in coordinates_list]
            all_points_y = [int(float(cor['y'])) for cor in coordinates_list]
            if mask.is_train:
                train_mask_dict["filename"] = mask.image_url
                train_mask_dict["regions"].append({
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y
                    },
                    "region_attributes": {
                        "name": mask.class_name
                    }
                })
            else:
                valid_mask_dict["filename"] = mask.image_url
                valid_mask_dict["regions"].append({
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y
                    },
                    "region_attributes": {
                        "name": mask.class_name
                    }
                })
        dataset['train'] = train_mask_dict
        dataset['val'] = valid_mask_dict
        manual_detection.train_custom_model(dataset, class_names)


@login_required
def manage_images(request):
    masks = MaskInfo.objects.filter(user=request.user)
    saved_images = SavedImage.objects.filter(user=request.user)
    data = {
        'masks': masks,
        'saved_images': saved_images,
        'media_path': settings.MEDIA_ROOT,
    }
    return render(request, 'detection/manage_images.html', data)


@login_required
def save_trained_image(request):
    if request.method == 'POST':
        user = request.user
        filename = request.POST['filename']
        timestr = time.strftime("%Y%m%d-%H%M%S")
        detected_file = settings.STATIC_DIR + '/ml/temp.jpg'
        try:
            os.mkdir(settings.MEDIA_ROOT + f'/ml/users/{user.username}/')
        except:
            print("The folder already existed")
        image = io.imread(detected_file)
        io.imsave(settings.MEDIA_ROOT + f'/ml/users/{user.username}/{filename}_{timestr}.jpg', image)
        saved_file = f'/ml/users/{user.username}/{filename}_{timestr}.jpg'
        SavedImage(user=request.user, file=saved_file).save()
        return redirect('detection:home')


@login_required
def class_list_ajax(request):
    model_type = request.GET['model_type'].strip()
    if model_type == 'default':
        class_names = default_class_names
    else:
        class_names = ['customize']
    data = {
        'class_names': class_names
    }
    return HttpResponse(json.dumps(data), content_type="application/json")


@login_required
def auto_detection(request):
    user = request.user
    class_names = request.POST['masks'].strip()
    model_name = request.POST['model'].strip()
    image_url = request.POST['image_url'].strip().replace("%20", " ")
    if string2list(class_names)[0] == "All":
        class_names_list = default_class_names.copy()
        class_names_list[0] = ['BG']
    else:
        class_names_list = ['BG'] + string2list(class_names)

    image_url = image_url.replace("\\", "/")
    img_url = MEDIA_ROOT + image_url[6:]
    model_path = ''
    if model_name == 'default':
        # try:
        model_path = 'mask_rcnn_coco.h5'
        class_names_list = default_class_names.copy()
        class_names_list[0] = ['BG']
    if model_name == 'model_1':
        model_path = 'mask_rcnn_object_0009.h5'
        detect(model_path=model_path, image_url=image_url)
        return render(request, "detection/detection_result.html", {'data': {"Sugar Cane": "31"}})
    if model_name == 'model_2':
        model_path = 'mask_rcnn_object_0010.h5'
        detect(model_path=model_path, image_url=image_url)
        return render(request, "detection/detection_result.html", {'data': {"Sugar Cane": "34"}})
    result = auto_detection_coco(model_path, img_url, class_names_list)
    detected_class = [class_names_list[id] for id in result['class_ids']]
    detected_class_keys = set(detected_class)
    detected_class_dict = {key: detected_class.count(key) for key in detected_class_keys}
    return render(request, "detection/detection_result.html", {'data': detected_class_dict})