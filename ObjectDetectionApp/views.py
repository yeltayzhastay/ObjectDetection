from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators import gzip
from ObjectML.object_detection import Detection
from .forms import VideoForm
from .models import Video
from pathlib import Path
import os

static_ML = Detection()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@gzip.gzip_page
def video_streaming(request):
    lastvideo = Video.objects.last()
    videofile = os.path.join(os.path.join(Path(__file__).resolve().parent.parent, 'media'), str(lastvideo)).replace('\\', '/')
    static_ML.cap_set(videofile)
    return StreamingHttpResponse(gen(static_ML), content_type="multipart/x-mixed-replace;boundary=frame")


def index(request):
    lastvideo = Video.objects.last()
    videofile = lastvideo
    print(videofile)
    form = VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    context = {
        'videofile': videofile,
        'form': form
        }
    return render(request, 'index.html', context=context)