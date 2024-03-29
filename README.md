# LLP_FLMm
Implementation for ACML 2021 paper **[A Two-Stage Training Framework with Feature-Label Matching
Mechanism for Learning from Label Proportions](https://proceedings.mlr.press/v157/yang21b.html)**

We use the code implemented by [HobbitLong](https://github.com/HobbitLong/SupContrast) to train the feature extractor.

Part of the code is borrowed from [kuangliu](https://github.com/kuangliu/pytorch-cifar).
Thank him for sharing the code. 

## How to run
1. Download the pretrained feature extractor weight from [here](https://drive.google.com/drive/folders/1d96DTXOdnI_MmgH1bR_fCdgU1swAZycD?usp=sharing)</br>
`(100_512_1000.pth: CIFAR100, batch_size 512, training epochs 1000, 10_512_1000: CIFAR10)` </br>
Or you can train it by your own following the guidence in [SupContrast](https://github.com/HobbitLong/SupContrast) </br>and make sure you take the SimCLR method instead of Supervised Contrastive learning method.
2. `python main.py --learning_rate 1 --packet_size 128  --ckpt pretrain_feature_extractor --dataset cifar100 --thre 0.01 --thre2 0.01 --acc_save_path acc.pkl --epochs 200`
3. As for `Ours without FLMm`, you can just comment out the following code in function `train()` in `main.py` </br>
to get a similar results.

```
        nll_loss = nn.NLLLoss()
        for r in range(int(1024/opt.packet_size)):
            if not threshold_label[r] or not batch_features[r]:
                continue
            for lal in threshold_label[r]:
                curt_bag_feature = batch_features[r]
                f = torch.stack(curt_bag_feature).cuda()
                tar = torch.tensor([lal for kkk in range(f.shape[0])],dtype=torch.long).cuda()
                output = classifier(f.detach())  # [1024,10]
                outputs = F.log_softmax(output,dim=-1)
                nll = 0.0005*nll_loss(outputs,tar)
                optimizer.zero_grad()
                nll.backward()
                optimizer.step()
```
