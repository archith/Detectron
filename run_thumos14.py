import os, sys
import multiprocessing as mp
from traceback import print_exc



video_img_root = '/media/ssd1/archith/video_analysis/thumos14/subsampled_images_original_fps/thumos14/'
viz_img_root = '/media/brain/archith/video_analysis/thumos14/full_fps_data/object_detection_results/detectron/visualization'
det_obj_root = '/media/brain/archith/video_analysis/thumos14/full_fps_data/object_detection_results/detectron/data'
gpu_list = ['0']
num_workers = 10 # number of workers per gpu

for x in [viz_img_root, det_obj_root]:
    if not os.path.isdir(x):
        os.makedirs(x)

command_fmt = "python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml \
    --image-ext jpg \
    --wts https://s3-us-west-2.amazonaws.com/detectron/36494496/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml.07_50_11.fkwVtEvg/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    --output-dir {} \
    --output-csv-file {} \
    --image-prefix=img_ \
    {}"

def get_free_gpu(gpu_queue):
  free_gpu = gpu_queue.get()
  return free_gpu

manager = mp.Manager()
available_gpu_queue = manager.Queue(maxsize=len(gpu_list)*num_workers)

for gpu in  gpu_list*num_workers:
  available_gpu_queue.put(gpu)

def exec_command(command):
    try:
        global available_gpu_queue

        dev_id = get_free_gpu(available_gpu_queue)
        command = 'CUDA_VISIBLE_DEVICES={} {}'.format(dev_id, command)
        os.system(command)

        available_gpu_queue.put(dev_id)

        return
    except Exception as e:
        print_exc()



commands = []

video_names = os.listdir(video_img_root)
video_names.sort()

for vid_name in video_names:
    vid_viz_dir = os.path.join(viz_img_root, vid_name)
    if not os.path.isdir(vid_viz_dir):
        os.makedirs(vid_viz_dir)

    vid_csv_file = os.path.join(det_obj_root, '{}.csv'.format(vid_name))

    vid_img_dir = os.path.join(video_img_root, vid_name)

    commands.append(command_fmt.format(vid_viz_dir, vid_csv_file, vid_img_dir))


p = mp.Pool(len(gpu_list)*num_workers)
p.map(exec_command, commands)

print "Finished!!"