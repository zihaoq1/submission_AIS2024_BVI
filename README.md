- This work is built on FasterVQA (https://github.com/VQAssessment/FAST-VQA-and-FasterVQA), trained with a novel ranking-based training strategy on multiple datasets.
- To test the method on single video, simply call:

```shell
run vqa.py -m FasterVQA -v /path/to/video.mp4
```

- You may also test the method on multiple videos by calling:

```shell
run vqa.py -m FasterVQA -v /path/to/video/directory -c /optional/ais24_pred.csv
```