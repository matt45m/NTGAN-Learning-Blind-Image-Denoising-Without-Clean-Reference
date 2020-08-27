import torch
import numpy as np
from network import NTGAN
import scipy.io as sio
import os
import h5py
from torch.autograd import Variable
import matplotlib.pyplot as plt


def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf


def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx, yy, bb]
    return sigma


def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma


def denoiser(Inoisy=None):
    net = NTGAN(im_ch=3, inter_ch=64, SA_ch=2, CA_ch=4)
    net.load_state_dict(torch.load('./models/ntgan.pth'))
    net.cuda()

    img_n = Inoisy.astype(np.float32).transpose(2, 0, 1)
    img_n = torch.from_numpy(img_n).unsqueeze(0)
    img_n = Variable(img_n.cuda(), requires_grad=False)
    nlm_t = Variable(torch.zeros_like(img_n).cuda(), requires_grad=False)
    with torch.no_grad():
        img_denoised = net(img_n, nlm_t)
    img_denoised = img_denoised.cpu().detach().numpy().squeeze()
    img_denoised = img_denoised.transpose(1, 2, 0).clip(0, 1)

    return img_denoised


def denoise_srgb(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:pass

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(0, 50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(0, 20):
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)

            Idenoised_crop = denoiser(Inoisy_crop)

            Idenoised_crop = np.float32(Idenoised_crop)

            crop_path = './DND_crops'
            crop_path = os.path.join(crop_path, '%04d_%02d.png' % (i + 1, k + 1))
            plt.imsave(crop_path, Idenoised_crop)

            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))


def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version = "1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat' % (i + 1, bb + 1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )


if __name__=='__main__':
    denoise_srgb(denoiser=denoiser, data_folder='./DND_eval', out_folder='./DND_subm')
    bundle_submissions_srgb(submission_folder='./DND_subm')

    print('Finished')





