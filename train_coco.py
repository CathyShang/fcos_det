# 导入模型
from model.fcos import FCOSDetector

import torch
from dataset.COCO_dataset import COCODataset
import math,time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from coco_eval import evaluate_coco, COCOGenerator
generator=COCOGenerator("/home/wangzy/shangzq/datasets/NWPU VHR-10 COCO/test","/home/wangzy/shangzq/datasets/NWPU VHR-10 COCO/annotations/test.json")

from model.config import DefaultConfig


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs") # 120
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")  # 8
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation") #4
    parser.add_argument("--n_gpu", type=str, default='1', help="number of cpu threads to use during batch generation")
    parser.add_argument("--save_path", type=str, default='test_0', help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()
    return opt

def main(opt):
    # todo 2022-3-23
    import torch.random as random
    import os
    from datetime import datetime
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(BASE_DIR, "..", "fcos_checkpoint")
    cfg = DefaultConfig()

    def make_logger(out_dir, model):
        """
        在out_dir文件夹下以当前时间命名 + 模型名称，创建日志文件夹，并创建logger用于记录信息
        :param out_dir: str
        :return:
        """
        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
        log_dir = os.path.join(out_dir, time_str + "_" + model)  # 根据config中的创建时间作为文件夹名
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 创建logger
        # path_log = os.path.join(log_dir, "log.log")
        # logger = Logger(path_log)
        # logger = logger.init_logger()
        # return logger, log_dir
        return log_dir

    log_dir = make_logger(res_dir, model=cfg.model_name)

    from tensorboardX import SummaryWriter
    logger = SummaryWriter(log_dir) # ()文件路径

    os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # random.seed(824)
    transform = Transforms()
    train_dataset=COCODataset("/home/wangzy/shangzq/datasets/NWPU VHR-10 COCO/train",
                              '/home/wangzy/shangzq/datasets/NWPU VHR-10 COCO/annotations/train.json',transform=transform)


    model=FCOSDetector(mode="training").cuda()
    model = torch.nn.DataParallel(model)
    BATCH_SIZE=opt.batch_size
    EPOCHS=opt.epochs
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate_fn,
                                             num_workers=opt.n_cpu,worker_init_fn = np.random.seed(0))
    steps_per_epoch=len(train_dataset)//BATCH_SIZE
    TOTAL_STEPS=steps_per_epoch*EPOCHS
    # warmup 学习率调整
    WARMUP_STEPS=500
    WARMUP_FACTOR = 1.0 / 3.0
    GLOBAL_STEPS=0
    LR_INIT= 0.001 # 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)
    lr_schedule = [120000, 160000]
    def lr_func(step):
        lr = LR_INIT
        if step < WARMUP_STEPS:
            alpha = float(step) / WARMUP_STEPS
            warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
            lr = lr*warmup_factor
        else:
            for i in range(len(lr_schedule)):
                if step < lr_schedule[i]:
                    break
                lr *= 0.1
        return float(lr)


    model.train()

    for epoch in range(EPOCHS):

        for epoch_step,data in enumerate(train_loader):
            # todo
            if epoch_step==0:
                iters_per_epoch = int(len(train_dataset) / len(data))

            batch_imgs,batch_boxes,batch_classes=data
            batch_imgs=batch_imgs.cuda()
            batch_boxes=batch_boxes.cuda()
            batch_classes=batch_classes.cuda()

            lr = lr_func(GLOBAL_STEPS)
            for param in optimizer.param_groups:
                param['lr']=lr

            start_time=time.time()

            optimizer.zero_grad()
            losses=model([batch_imgs,batch_boxes,batch_classes])
            loss=losses[-1]
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),3)
            optimizer.step()

            end_time=time.time()
            cost_time=int((end_time-start_time)*1000)
            if (epoch_step+1)%10==0:
                print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f"%\
                (GLOBAL_STEPS,epoch+1,epoch_step+1,steps_per_epoch,losses[0].mean(),losses[1].mean(),losses[2].mean(),cost_time,lr, loss.mean()))

            info = {
                'loss': loss.mean(),
                'loss_cls': losses[0].mean(),
                'cnt_loss': losses[1].mean(),
                'loss_box': losses[2].mean()
            }
            logger.add_scalars("logs_s_{}/losses".format("train"), info, epoch * iters_per_epoch + epoch_step)

            GLOBAL_STEPS+=1

        if (epoch+1)%10==0 or epoch==EPOCHS:

            torch.save(model.state_dict(), log_dir + "/model_{}.pth".format(epoch+1))

    # 关闭logger
    logger.close()
    
if __name__=="__main__":
    opt = parse_config()
    main(opt)
