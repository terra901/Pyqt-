# -*- coding: utf-8 -*-
# @Time : 2022/12/13 16:02
# @Author : Sorrow
# @File : evaltest.py
# @Software: PyCharm
from DL_Model.pingjia import SegmentationMetric


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    beset_miou = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    train_cpa = []
    val_cpa = []
    min_loss = np.inf
    min_miou = 0
    min_cpa = 0
    min_recall = 0
    best = 0
    decrease = 1
    not_improve = 0
    train_miou = []
    val_miou = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        cpa = 0
        miou1 = 0
        recall = 0
        f1 = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            # print("image_tiles",image_tiles.shape)
            # print("mask_tiles", mask_tiles.shape)

            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)
            # forward
            # print("image",image.size())
            image = image_tiles.to(device)
            mask = mask_tiles.to(device)

            output = model(image)

            # loss
            loss = criterion(output,mask)
            # evaluation metrics
            metric = SegmentationMetric(2)  # ()里面表示分类
            metric.addBatch(output, mask_tiles)
            cpa += metric.meanPixelAccuracy()
            miou1 += metric.meanIntersectionOverUnion()
            recall += metric.recall()
            f1 += metric.F1Score()
            accuracy += metric.pixelAccuracy()

            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            val_cpa_score = 0
            test_miou = 0
            val_Recall = 0
            val_F1 = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data
                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)

                    output = model(image)
                    output2 = output.data.cpu().numpy()
                    # loss
                    loss = criterion(output, mask)


                    test_loss += loss.item()

                    metric = SegmentationMetric(2)
                    metric.addBatch(output, mask_tiles)
                    val_cpa_score += metric.meanPixelAccuracy()
                    test_miou += metric.meanIntersectionOverUnion()
                    val_Recall += metric.recall()
                    val_F1 += metric.F1Score()
                    test_accuracy += metric.pixelAccuracy()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))
# 保存精确率最高的权重


            if val_cpa_score / len(val_loader) > min_cpa:
                min_cpa = val_cpa_score / len(val_loader)
                torch.save(model.state_dict(), "F:\\unet2\\weight\\focal2_0.25\\" + "best_cpa1.pth")
                torch.save(model, "F:\\unet2\\weight\\focal2_0.25\\" + "best_cpa1.pt")
                print("best cpa has saved:{:.3f} --- > {:.3f}".format(min_cpa, (val_cpa_score / len(val_loader))))

            if val_Recall / len(val_loader) > min_recall:
                min_recall = val_Recall / len(val_loader)
                torch.save(model.state_dict(), "F:\\unet2\\weight\\focal2_0.25\\" + "best_recall1.pth")
                torch.save(model, "F:\\unet2\\weight\\focal2_0.25\\" + "best_recall1.pt")
                print("best recall has saved:{:.3f} --- > {:.3f}".format(min_recall, (val_Recall / len(val_loader))))

            train_cpa.append(cpa / len(train_loader))
            val_cpa.append(val_cpa_score / len(val_loader))
            train_miou.append(miou1 / len(train_loader))
            val_miou.append(test_miou / len(val_loader))
            train_recall.append(recall / len(train_loader))
            val_recall.append(val_Recall / len(val_loader))
            train_f1.append(f1 / len(train_loader))
            val_f1.append(val_F1 / len(val_loader))

            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.4f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.4f}..".format(test_loss / len(val_loader)),
                  "train_cpa:{:.4f}..".format(cpa / len(train_loader)),
                  "val_cpa:{:.4f}..".format(val_cpa_score / len(val_loader)),
                  "train_miou:{:.4f}..".format(miou1 / len(train_loader)),
                  "val_miou:{:.4f}..".format(test_miou / len(val_loader)),
                  "train_recall:{:.4f}..".format(recall / len(train_loader)),
                  "val_recall:{:.4f}..".format(val_Recall / len(val_loader)),
                  "train_f1:{:.4f}..".format(f1 / len(train_loader)),
                  "val_f1:{:.4f}..".format(val_F1 / len(val_loader)),
                  "Time: {:.4f}m".format((time.time() - since) / 60))
     # 每隔50轮保存一次权重
        if e % 50 == 0:
            print('saving model...')
            torch.save(model.state_dict(), "F:\\unet2\\weight\\focal2_0.25\\" + "unet" + "%03d" % (e) + ".pth")
            torch.save(model, "F:\\unet2\\weight\\focal2_0.25\\" + "UNet" + "%03d" % (e) + ".pt")

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_cpa': train_cpa, 'val_cpa': val_cpa,
               'train_miou1': train_miou, 'val_miou1': val_miou,
               'train_recall': train_recall, 'val_recall': val_recall,
               'train_f1': train_f1, 'val_f1': val_f1,
               'lrs': lrs}

    print('Total time: {:.3f} m'.format((time.time() - fit_time) / 60))
    return history
