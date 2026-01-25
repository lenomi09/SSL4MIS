import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../data/ACDC")
parser.add_argument(
    "--exp", type=str, default="ACDC/Uncertainty_Rectified_Pyramid_Consistency"
)
parser.add_argument("--model", type=str, default="unet_urpc")
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--labeled_num", type=int, default=7)


def calculate_metric_percase(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 100.0, 100.0
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), "r")
    image = h5f["image"][:]
    label = h5f["label"][:]
    prediction = np.zeros_like(label)

    net.eval()
    with torch.no_grad():
        for ind in range(image.shape[0]):
            slice = image[ind]
            x, y = slice.shape
            slice = zoom(slice, (256 / x, 256 / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            out = net(input)
            if isinstance(out, (tuple, list)):
                out = out[0]  # URPC: lấy output chính

            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    metrics = []
    for cls in [1, 2, 3]:
        metrics.append(calculate_metric_percase(prediction == cls, label == cls))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return metrics


def Inference(FLAGS):
    with open(FLAGS.root_path + "/test.list", "r") as f:
        image_list = f.readlines()
    image_list = sorted([item.replace("\n", "").split(".")[0] for item in image_list])

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model
    )
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model
    )
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(
        net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes
    ).cuda()
    save_mode_path = os.path.join(
        snapshot_path, "{}_best_model.pth".format(FLAGS.model)
    )
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    metric_sum = np.zeros((3, 3))
    for case in tqdm(image_list):
        case_metrics = test_single_volume(case, net, test_save_path, FLAGS)
        metric_sum += np.array(case_metrics)

    avg_metric = metric_sum / len(image_list)
    return avg_metric


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print("Mean:", metric.mean(axis=0))
